import copy,json
import joblib,numpy as np,pandas as pd,pytest
torch=pytest.importorskip('torch')
from scipy import sparse as sp
from .data import add_chunk
from .features import diffusion
from .model import TemporalMixer,pretrain,encode
from .finetune import VotingTask


def empty(n=3):
    from collections import Counter
    s={k:sp.csr_matrix(shape,dtype=float) for k,shape in [('counts',(n,n)),('bytes',(n,n)),('peers',(4*n,8192)),('services',(4*n,4096)),('protocols',(4*n,128))]}
    s.update(quarter_counts=[sp.csr_matrix((n,n),dtype=float) for _ in range(4)],quarter_bytes=[sp.csr_matrix((n,n),dtype=float) for _ in range(4)],numeric=np.zeros((4*n,18)),events=np.zeros(n),contacts=[Counter() for _ in range(n)],discarded_quarter_straddles=0,hosts=['a','b','c'][:n])
    return s


def test_directional_volumes_and_external_peer():
    s=empty();f=pd.DataFrame({'src':['a','b','external','a'],'dst':['b','a','a','a'],
        'bytes':[100,30,70,1000],'resp_bytes':[20,5,9,1000],'packets':[10,3,7,100],'resp_packets':[2,1,1,100],
        'ts':[1,2,3,4],'duration':[0,0,0,0],'proto':['6']*4,'dst_port':['80']*4})
    add_chunk(s,f,pd.Index(['a','b','c']),0,3600)
    # Для a: TX=100+5+9, RX=20+30+70; для b наоборот. Обе роли учтены.
    np.testing.assert_array_equal(s['numeric'][0,[0,1]]+s['numeric'][0,[9,10]],[114,120])
    np.testing.assert_array_equal(s['numeric'][1,[0,1]]+s['numeric'][1,[9,10]],[50,105])
    assert s['events'].tolist()==[3,2,0]
    assert s['bytes'][0,1]==105 and s['bytes'][1,0]==50
    assert s['contacts'][0]['external']==1


def test_quarter_causality_and_full_hour_counts():
    s=empty();f=pd.DataFrame({'src':['a','a'],'dst':['b','b'],'bytes':[100,200],'resp_bytes':[2,4],
        'packets':[10,20],'resp_packets':[1,2],'ts':[899,900],'duration':[2,0],'proto':['6','6'],'dst_port':['80','80']})
    add_chunk(s,f,pd.Index(['a','b','c']),0,3600)
    assert s['counts'][0,1]==2 and s['quarter_counts'][0].nnz==0
    assert s['quarter_counts'][1][0,1]==1 and s['numeric'][0].sum()==0
    assert s['events'][0]==2 and s['discarded_quarter_straddles']==1


def test_graph_is_equivariant_to_node_relabeling():
    n=8;rng=np.random.default_rng(1);s={'hosts':list(map(str,range(n)))}
    s['quarter_counts']=[sp.csr_matrix(rng.integers(0,3,(n,n))) for _ in range(4)];s['quarter_bytes']=[a*20 for a in s['quarter_counts']]
    x=rng.normal(size=(4*n,5)).astype(np.float32);p=rng.permutation(n)
    other={'hosts':s['hosts'],'quarter_counts':[a[p][:,p] for a in s['quarter_counts']],'quarter_bytes':[a[p][:,p] for a in s['quarter_bytes']]}
    expected=diffusion(s,x).reshape(4,n,-1)[:,p]
    actual=diffusion(other,x.reshape(4,n,-1)[:,p].reshape(4*n,5)).reshape(4,n,-1)
    np.testing.assert_allclose(actual,expected,atol=1e-6)


def test_mixer_no_id_input_and_own_branch_is_fixed():
    torch.manual_seed(1);m=TemporalMixer(dim=8);x=torch.randn(20,4,4,8)
    z=m.embedding(x);loss=z[:,8:].square().sum();loss.backward()
    before=z[:,:8]/z[:,:8].norm(dim=1,keepdim=True)
    with torch.no_grad():
        for p in m.parameters():p.add_(torch.randn_like(p)*.01)
        after=m.embedding(x);after=after[:,:8]/after[:,:8].norm(dim=1,keepdim=True)
    torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-6)
    p=torch.randperm(20);torch.testing.assert_close(m.embedding(x[p]),m.embedding(x)[p])


def test_future_loss_updates_shared_encoder_without_future_inputs():
    m=TemporalMixer(dim=8);x=torch.randn(10,4,4,8)
    patch=m.input(x[:,:1].flatten(2));latent=torch.cat([m.stable(patch),m.dynamic(patch)],2)
    loss=(m.next_patch(latent)-x[:,1:2].flatten(2)).square().mean();loss.backward()
    assert m.input.weight.grad.abs().sum()>0 and m.stable.weight.grad.abs().sum()>0


def test_training_and_reload(tmp_path):
    rng=np.random.default_rng(2);x=[rng.normal(size=(20,4,4,8)).astype(np.float32) for _ in range(3)];active=[np.ones(20,bool)]*3
    m,log=pretrain(x,active,steps=2,dim=8)
    assert np.isfinite(encode(m,x[0])).all()
    path=tmp_path/'m.pt';torch.save({'state':m.state_dict(),'mode':m.mode,'dim':m.dim,'hidden':m.hidden},path)
    from .evaluate import checkpoint
    np.testing.assert_array_equal(encode(m,x[0]),encode(checkpoint(path),x[0]))


def test_hard_voting_forward_matches_sparse_source_counts():
    rng=np.random.default_rng(4);n=16;hosts=[f'h{i}' for i in range(n)]
    old=[{f'p{i%4}':i+1} for i in range(n)]
    target={'events':np.full(n,400),'contacts':[{f'p{(i+1)%4}':10} for i in range(n)]}
    task=VotingTask(old,target,hosts);z=torch.tensor(rng.random((n,12)),dtype=torch.float32,requires_grad=True);z=torch.nn.functional.normalize(z,dim=1)
    loss,votes=task.loss(z,np.array([0,1]));sim=(z.detach()@z.detach().T).numpy()
    for row,q in enumerate([0,1]):
        sim[q,q]=-1e9;top=np.argsort(-sim[q])[:10]
        expected=np.asarray(task.matrix[top].T@sim[q,top].clip(min=0)).ravel()
        np.testing.assert_allclose(votes.detach().numpy()[row],expected,rtol=1e-6,atol=1e-4)
    loss.backward();assert torch.isfinite(loss)


def test_pooled_history_excludes_absent_hosts_and_preserves_source_behavior():
    from .model import pooled_source
    from .features import unit
    rng=np.random.default_rng(10);history=[rng.normal(size=(20,4,4,8)).astype(np.float32) for _ in range(3)]
    m=TemporalMixer(dim=8);presence=[np.ones(20,bool),np.zeros(20,bool),np.ones(20,bool)]
    a=pooled_source(m,history,[2,3,4],presence=presence)
    changed=[h.copy() for h in history];changed[1]*=100
    b=pooled_source(m,changed,[2,3,4],presence=presence)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(unit(a[:,:8]),unit(history[0][:,:,0,:].mean(1)),atol=1e-6)
    # Размер query-когорты может меняться; lookup собственного ID не нужен.
    assert encode(m,history[0][:3]).shape==(3,392)


def test_global_gallery_training_uses_shared_encoder():
    from .continue_identity import train_full_gallery
    rng=np.random.default_rng(3);x=[rng.normal(size=(20,4,4,8)).astype(np.float32) for _ in range(3)]
    m=TemporalMixer(dim=8);before=m.input.weight.detach().clone()
    trained,_=train_full_gallery(m,x,[np.ones(20,bool)]*3,steps=2)
    assert not torch.equal(before,trained.input.weight)
    assert np.isfinite(encode(trained,x[0])).all()


def test_no_graph_control_really_removes_neighbor_input():
    torch.manual_seed(21);x=torch.randn(10,4,4,8);changed=x.clone();changed[:,:,3,:]*=30
    m=TemporalMixer(dim=8,mode='no_graph')
    torch.testing.assert_close(m.embedding(x),m.embedding(changed))
    graph=TemporalMixer(dim=8,mode='graph')
    assert not torch.allclose(graph.embedding(x),graph.embedding(changed))


def test_config_rejects_future_training_and_overlap(tmp_path):
    from .config import load
    base={'train':[dict(name='day02',day=2,start=100,end=200)],'development':dict(name='day13',day=13,start=300,end=400),'benchmark':[]}
    path=tmp_path/'config.json';path.write_text(json.dumps(base));assert load(path)==base
    base['train'].append(dict(name='forbidden',day=14,start=210,end=220));path.write_text(json.dumps(base))
    with pytest.raises(ValueError,match='after day12'):load(path)
    base['train']=base['train'][:1];base['development']['start']=150;path.write_text(json.dumps(base))
    with pytest.raises(ValueError,match='overlaps'):load(path)


def test_own_profile_keeps_diversity_without_peer_id_lookup():
    from .features import own_raw
    a=empty();a['window']={'start':0,'end':3600}
    a['peers'][0,0]=10;a['peers'][0,1]=10;a['services'][0,0]=20
    original=own_raw(a,2)
    b=copy.deepcopy(a);b['peers']=a['peers'][:,np.roll(np.arange(8192),20)]
    np.testing.assert_array_equal(original,own_raw(b,2))
    c=copy.deepcopy(a);c['peers']=sp.csr_matrix(c['peers'].shape);c['peers'][0,0]=20
    assert not np.array_equal(original,own_raw(c,2))


def test_training_rejects_stale_encoded_rows_after_transform_change(tmp_path):
    from .fit import verify_training
    from .data import sha
    (tmp_path/'transforms.joblib').write_bytes(b'frozen-vocabulary')
    np.save(tmp_path/'day02.npy',np.eye(3))
    meta={'transforms_sha256':sha(tmp_path/'transforms.joblib'),
          'train_feature_sha256':{'day02':sha(tmp_path/'day02.npy')}}
    verify_training(tmp_path,meta)
    np.save(tmp_path/'day02.npy',np.eye(3)[::-1])
    with pytest.raises(ValueError,match='кэш признаков'):verify_training(tmp_path,meta)
    np.save(tmp_path/'day02.npy',np.eye(3))
    (tmp_path/'transforms.joblib').write_bytes(b'changed-vocabulary')
    with pytest.raises(ValueError,match='словарь'):verify_training(tmp_path,meta)
