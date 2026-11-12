import * as fontkit from "fontkit";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const font = fontkit.openSync(path.join(root, "fonts/pt-display.ttf"));
const chars = [
  ...new Set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя-–.,:!?/+()",
  ),
];
const glyphs = {};
for (const char of chars) {
  const glyph = font.glyphsForString(char)[0];
  let outline = [],
    start = [0, 0];
  for (const { command, args: a } of glyph.path.commands) {
    if (command === "moveTo") {
      outline.push("m", ...a);
      start = a;
    } else if (command === "lineTo") outline.push("l", ...a);
    else if (command === "quadraticCurveTo")
      outline.push("q", a[2], a[3], a[0], a[1]);
    else if (command === "bezierCurveTo")
      outline.push("b", a[4], a[5], a[0], a[1], a[2], a[3]);
    else if (command === "closePath") outline.push("l", ...start);
  }
  const b = glyph.bbox;
  glyphs[char] = {
    ha: glyph.advanceWidth,
    x_min: Number.isFinite(b.minX) ? b.minX : 0,
    x_max: Number.isFinite(b.maxX) ? b.maxX : 0,
    o: outline
      .map((x) => (typeof x === "number" ? Math.round(x * 100) / 100 : x))
      .join(" "),
  };
}
fs.writeFileSync(
  path.join(root, "fonts/PTDisplay_Regular.json"),
  JSON.stringify({
    glyphs,
    familyName: "PT Display / Oswald",
    ascender: font.ascent,
    descender: font.descent,
    underlinePosition: -100,
    underlineThickness: 50,
    boundingBox: {
      xMin: font.bbox.minX,
      xMax: font.bbox.maxX,
      yMin: font.descent,
      yMax: font.ascent,
    },
    resolution: font.unitsPerEm,
    original_font_information: {
      name: "Oswald 600",
      license: "SIL Open Font License; see pt-display-OFL.txt",
    },
  }),
);
console.log("Generated", Object.keys(glyphs).length, "3D glyphs");
