# temp script: convert SVG fill=/stroke= presentation attrs to style="" with theme vars
import re, io

PATH = r"D:\ai-projects\NCP-AAI\visuals\domain-10-human-ai-interaction.html"
MARK = "<!-- ============ THREE.JS SCENE ============ -->"

CMAP = {
    "#76b900": "var(--green)",
    "#a8d36a": "var(--green)",
    "#b8e07a": "var(--green)",
    "#d6f3a1": "var(--green)",
    "#e05252": "var(--red)",
    "#f0a0a0": "var(--red)",
    "#e8a33d": "var(--amber)",
    "#ecc185": "var(--amber)",
    "#4da3ff": "var(--blue)",
    "#9ec9f5": "var(--blue)",
    "#2c4a73": "var(--blue)",
    "#b07aff": "var(--purple)",
    "#c9b3ee": "var(--purple)",
    "#8fa1b8": "var(--muted)",
    "#dde6f2": "var(--text)",
    "#121a29": "var(--bg3)",
    "#1a2030": "var(--panel2)",
    "#0a1220": "var(--bg2)",
    "#1f2c42": "var(--line)",
    "#1d1420": "color-mix(in srgb, var(--red) 12%, var(--bg2))",
    "#10220c": "color-mix(in srgb, var(--green) 12%, var(--bg2))",
    "#221a0c": "color-mix(in srgb, var(--amber) 12%, var(--bg2))",
    "#0d1726": "color-mix(in srgb, var(--blue) 12%, var(--bg2))",
    "rgba(224,82,82,.45)": "color-mix(in srgb, var(--red) 45%, transparent)",
    "rgba(224,82,82,.32)": "color-mix(in srgb, var(--red) 32%, transparent)",
    "rgba(232,163,61,.32)": "color-mix(in srgb, var(--amber) 32%, transparent)",
    "rgba(118,185,0,.55)": "color-mix(in srgb, var(--green) 55%, transparent)",
    "rgba(118,185,0,.4)": "color-mix(in srgb, var(--green) 40%, transparent)",
}

with io.open(PATH, "r", encoding="utf-8") as f:
    content = f.read()

idx = content.index(MARK)
head, tail = content[:idx], content[idx:]

count = [0]
attr_re = re.compile(r'\s(fill|stroke)="([^"]+)"')
tag_re = re.compile(r'<(rect|line|path|polygon|polyline|circle|ellipse|text|tspan|g)\b[^>]*?/?>')

def fix_tag(m):
    tag = m.group(0)
    props = []
    def take(am):
        name, val = am.group(1), am.group(2)
        if val in CMAP:
            props.append("%s:%s" % (name, CMAP[val]))
            count[0] += 1
            return ""          # remove the attr
        return am.group(0)     # keep (e.g. fill="none")
    new = attr_re.sub(take, tag)
    if not props:
        return tag
    style_add = ";".join(props)
    sm = re.search(r'style="([^"]*)"', new)
    if sm:
        merged = sm.group(1).rstrip(";") + ";" + style_add
        new = new[:sm.start()] + 'style="%s"' % merged + new[sm.end():]
    else:
        if new.endswith("/>"):
            new = new[:-2].rstrip() + ' style="%s"/>' % style_add
        else:
            new = new[:-1].rstrip() + ' style="%s">' % style_add
    return new

head = tag_re.sub(fix_tag, head)

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(head + tail)

print("SVG attr replacements:", count[0])
# any leftover raw hex fills/strokes in the head region?
left = set(re.findall(r'(?:fill|stroke)="(#[0-9a-fA-F]{3,6}|rgba?\([^"]*\))"', head))
print("leftover attr colors:", sorted(left))
