"""Pair a Telegram bot to @physcorgi and send deployment notifications."""

import json
import os
import re
import sys
from pathlib import Path
from string import Template
from getpass import getpass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

OWNER = "physcorgi"


def telegram(token, method, fields):
    if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token):
        raise ValueError("Invalid bot token format")
    request = Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=json.dumps(fields).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise ValueError("Telegram rejected the request")
    return result["result"]


def private_chat(chat_id):
    if not re.fullmatch(r"[1-9][0-9]{0,18}", chat_id):
        raise ValueError("Expected the numeric ID of the paired private chat")
    return int(chat_id)


def pair():
    token = getpass("BotFather token: ").strip()
    matches = set()
    for update in telegram(token, "getUpdates", {"allowed_updates": ["message"]}):
        message = update.get("message", {})
        chat, sender = message.get("chat", {}), message.get("from", {})
        if (chat.get("type") == "private" and chat.get("id") == sender.get("id")
                and sender.get("username", "").lower() == OWNER
                and message.get("text", "").split(maxsplit=1)[:1] == ["/start"]):
            matches.add(private_chat(str(chat["id"])))
    if len(matches) != 1:
        raise ValueError("Send /start privately from @physcorgi, then retry pairing")
    print(matches.pop())


def notify(message):
    token = os.getenv("PT_TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("PT_TELEGRAM_CHAT_ID")
    if not token and not chat_id:
        print("Telegram notifications are not configured.")
        return
    if not token or not chat_id:
        raise ValueError("Set both Telegram secrets")
    run_id = os.environ["GITHUB_RUN_ID"]
    if not run_id.isdigit():
        raise ValueError("Invalid GitHub run ID")
    url = f"https://github.com/CorgCode/pt-project/actions/runs/{run_id}"
    telegram(token, "sendMessage", {
        "chat_id": private_chat(chat_id), "text": message,
        "reply_markup": {"inline_keyboard": [[{"text": "Open in GitHub", "url": url}]]},
    })


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["pair"]:
            pair()
        elif sys.argv[1:] == ["alertmanager-config"]:
            chat_id = private_chat(os.getenv("PT_TELEGRAM_CHAT_ID", ""))
            template = Path(__file__).with_name("alertmanager.yml.template").read_text()
            print(Template(template).substitute(PT_TELEGRAM_CHAT_ID=chat_id), end="")
        elif len(sys.argv) == 3 and sys.argv[1] == "notify":
            notify(sys.argv[2])
        else:
            raise SystemExit("Usage: telegram.py pair | alertmanager-config | notify MESSAGE")
    except HTTPError as error:
        reasons = {
            401: "Telegram rejected the token. Use the current token from BotFather.",
            404: "Telegram did not recognize this bot token.",
            409: "Telegram cannot read updates while a webhook or another poller is active.",
            429: "Telegram is rate limiting requests. Wait briefly and retry.",
        }
        print(reasons.get(error.code, f"Telegram returned HTTP {error.code}."), file=sys.stderr)
        raise SystemExit(1) from None
    except ValueError as error:
        # Pairing errors use fixed messages. Never print a Telegram payload or URL.
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
    except OSError:
        print("Telegram request failed.", file=sys.stderr)
        raise SystemExit(1) from None
