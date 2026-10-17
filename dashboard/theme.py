"""Look and feel: a draughting-film plane, ink text, and one verdict colour.

Palette notes
  plane   cool pale grey, like drawing film       ink     blue-black, never pure black
  teal    reserved for "adopted / confirmed"      grey    "rejected" is a normal, quiet outcome
  groups  the first five slots of a validated colour-blind-safe categorical palette
Text never wears a data colour: a coloured swatch sits beside the words instead.
"""

# validated with the dataviz palette checker: lightness band, chroma, colour-blind separation of neighbours
GROUP_COLORS = {
    "customer": "#2a78d6",
    "usage": "#eb6834",
    "recharge": "#1baf7a",
    "network": "#eda100",
    "risk": "#e87ba4",
}
FALLBACK_GROUP = "#8896a0"

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;1,6..72,400"
    "&family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');"
)

CSS = """
:root{
  --plane:#ECEFEE; --wash:#E2E7E8; --rule:#C5CDD2;
  --ink:#12212C; --ink2:#455663; --ink3:#5E6E7A;
  --teal:#0B7A6B; --teal-wash:#D3E8E3; --grey:#5E6E7A;
  --sans:'Schibsted Grotesk',ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;
  --serif:'Newsreader',Georgia,'Times New Roman',serif;
}
html,body,.stApp{background:var(--plane)!important;color:var(--ink);font-family:var(--sans)}
header[data-testid="stHeader"],footer:not(.foot),#MainMenu,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{max-width:1180px!important;padding:64px 36px 120px!important}
.stHtml,.stMarkdown{font-family:var(--sans)}

/* masthead */
.mast{display:grid;grid-template-columns:200px minmax(0,1fr);column-gap:48px;row-gap:0;align-items:baseline;padding-bottom:44px}
.mast .name{font-size:15px;font-weight:700;letter-spacing:-.01em}
.mast h1{font-size:34px;line-height:1.12;font-weight:600;letter-spacing:-.025em;margin:0;max-width:22ch;color:var(--ink)}
.mast p{grid-column:2;margin:18px 0 0;font-size:16px;line-height:1.5;color:var(--ink2);max-width:60ch}

/* section scaffold: a narrow rail that says what the section is, content on the right */
.sec{display:grid;grid-template-columns:200px minmax(0,1fr);gap:48px;padding:40px 0 52px;border-top:1px solid var(--rule)}
.rail h2{font-size:15px;font-weight:650;letter-spacing:-.005em;margin:0 0 8px}
.rail p{font-size:13.5px;line-height:1.5;color:var(--ink2);margin:0;max-width:24ch}
.body{min-width:0}

/* hero */
.headline{display:flex;gap:36px;align-items:flex-end;flex-wrap:wrap;margin-bottom:34px}
.big{font-size:88px;line-height:.9;font-weight:600;letter-spacing:-.045em}
.lede{margin:0;font-size:17px;line-height:1.5;color:var(--ink);max-width:46ch}
.lede span{color:var(--ink2)}
.ruler{position:relative;margin-top:6px}
.rrow{display:grid;grid-template-columns:190px minmax(0,1fr);gap:24px;align-items:center;min-height:56px}
.rname{font-size:14px;font-weight:600;line-height:1.3}
.rname small{display:block;font-weight:400;color:var(--ink3);font-size:12.5px}
.track{position:relative;height:56px}
.track .g{position:absolute;top:0;bottom:0;width:1px;background:var(--rule)}
.track .ci{position:absolute;top:50%;height:4px;margin-top:-2px;border-radius:2px;background:var(--ink)}
.track .ci::before,.track .ci::after{content:"";position:absolute;top:-5px;width:2px;height:14px;background:inherit;border-radius:1px}
.track .ci::before{left:0}.track .ci::after{right:0}
.track .dot{position:absolute;top:50%;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;background:var(--ink);border:2px solid var(--plane);box-sizing:content-box}
.track .val{position:absolute;top:50%;transform:translateY(-135%);margin-left:12px;font-size:14px;font-weight:650;font-variant-numeric:tabular-nums;white-space:nowrap}
.rrow.base .ci,.rrow.base .dot{background:var(--grey)}
.cw{position:absolute;top:0;bottom:30px;left:214px;right:0;pointer-events:none}
.chance{position:absolute;top:0;bottom:0;width:0;border-left:2px solid var(--ink);opacity:.9}
.chance span{position:absolute;top:-2px;left:8px;font-size:12.5px;color:var(--ink2);white-space:nowrap;transform:translateY(-100%)}
.axis{display:grid;grid-template-columns:190px minmax(0,1fr);gap:24px}
.axis .scale{position:relative;height:26px}
.axis .scale span{position:absolute;top:6px;transform:translateX(-50%);font-size:12px;color:var(--ink3);font-variant-numeric:tabular-nums}

/* ledger: the forest plot of every experiment */
.ledger,.mem,.checks{list-style:none!important;margin-left:0!important;margin-right:0!important;padding:0!important}
.lhead,.lrow{display:grid;grid-template-columns:28px minmax(240px,1fr) clamp(220px,34%,330px) 124px;gap:20px;align-items:stretch}
.lhead{align-items:end;padding-bottom:10px}
.lhead .scale{position:relative;height:34px}
.lhead .scale span{position:absolute;bottom:0;transform:translateX(-50%);font-size:12px;color:var(--ink3);font-variant-numeric:tabular-nums;white-space:nowrap}
.lhead .scale .bar{color:var(--teal);font-weight:600;transform:none;bottom:18px;font-size:12.5px;font-variant-numeric:normal}
.lrow{padding:16px 0;border-top:1px solid var(--rule)}
.lrow:hover{background:var(--wash)}
.idx{font-size:13px;color:var(--ink3);font-variant-numeric:tabular-nums;padding-top:3px}
.hyp{font-family:var(--serif);font-size:17.5px;line-height:1.38;margin:0;color:var(--ink)}
.why{display:block;margin-top:6px;font-family:var(--sans);font-size:13px;line-height:1.4;color:var(--ink3)}
.verdict{font-weight:650;color:var(--ink2)}
.lrow.adopt .verdict{color:var(--teal)}
.plot{position:relative;min-height:44px}
.plot .zone{position:absolute;top:-16px;bottom:-16px;right:0;background:var(--teal-wash)}
.plot .zero{position:absolute;top:-16px;bottom:-16px;width:0;border-left:1.5px solid var(--ink)}
.plot .ci{position:absolute;top:50%;height:2px;margin-top:-1px;background:var(--grey)}
.plot .ci::before,.plot .ci::after{content:"";position:absolute;top:-5px;width:2px;height:12px;background:inherit}
.plot .ci::before{left:0}.plot .ci::after{right:0}
.plot .dot{position:absolute;top:50%;width:8px;height:8px;margin:-4px 0 0 -4px;border-radius:50%;background:var(--grey);border:2px solid var(--plane);box-sizing:content-box}
.lrow:hover .dot{border-color:var(--wash)}
.lrow.adopt .ci{background:var(--teal)}
.lrow.adopt .dot{background:var(--teal)}
.nums{display:flex;flex-direction:column;justify-content:center;font-variant-numeric:tabular-nums;text-align:right}
.nums strong{font-size:16px;font-weight:650}
.nums span{font-size:12.5px;color:var(--ink3);white-space:nowrap}
.tally{margin:26px 0 0;font-size:15px;line-height:1.5;color:var(--ink2);max-width:70ch}
.tally b{color:var(--ink);font-weight:650}

/* signal and drift */
.share{display:flex;gap:2px;height:22px;border-radius:4px;overflow:hidden}
.share i{display:block;min-width:3px}
.keys{display:flex;flex-wrap:wrap;gap:8px 28px;margin:14px 0 30px;font-size:14px}
.key{display:inline-flex;align-items:center;gap:8px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;flex:none}
.key b{font-weight:600}.key span{color:var(--ink2);font-variant-numeric:tabular-nums}
.bars{display:grid;gap:10px}
.brow{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr) 64px;gap:18px;align-items:center;font-size:14px}
.brow .nm{display:flex;align-items:center;gap:10px;min-width:0}
.brow .nm em{font-style:normal;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.brow .tr{position:relative;height:14px}
.brow .tr b{position:absolute;left:0;top:0;height:14px;border-radius:0 4px 4px 0;min-width:2px}
.brow .tr .g{position:absolute;top:-4px;bottom:-4px;width:0;border-left:1px solid var(--rule)}
.brow .v{text-align:right;color:var(--ink2);font-variant-numeric:tabular-nums}
.gh{display:grid;grid-template-columns:minmax(0,340px) minmax(0,1fr) 64px;gap:18px;font-size:12px;color:var(--ink3);margin-bottom:6px}
.gh .scale{position:relative;height:18px}
.gh .scale span{position:absolute;white-space:nowrap}
.checks{margin-bottom:34px!important;display:grid;gap:10px}
.checks li{display:flex;gap:14px;align-items:baseline;font-size:15px;line-height:1.45}
.checks .when{width:112px;white-space:nowrap;flex:none;font-weight:600;font-variant-numeric:tabular-nums}
.pill{display:inline-block;font-size:12.5px;font-weight:650;padding:1px 9px;border-radius:99px;margin-right:8px;background:var(--wash);color:var(--ink2)}
.pill.ok{background:var(--teal-wash);color:var(--teal)}
.pill.warn{background:#F6DDD3;color:#9B3A12}
.note{margin:22px 0 0;font-size:13.5px;line-height:1.5;color:var(--ink3);max-width:70ch}

/* memory */
.mem li{padding:16px 0;border-top:1px solid var(--rule)}
.mem li:first-child{border-top:0;padding-top:4px}
.mem p{font-family:var(--serif);font-size:17.5px;line-height:1.42;margin:0;max-width:62ch}
.mem small{display:block;margin-top:6px;font-size:12.5px;color:var(--ink3)}
.empty{font-size:15px;line-height:1.5;color:var(--ink2);max-width:60ch}
.empty code{font-family:var(--sans);font-weight:600;background:var(--wash);padding:1px 6px;border-radius:4px}
.foot{padding-top:28px;border-top:1px solid var(--rule);font-size:12.5px;line-height:1.6;color:var(--ink3)}

/* the search box is a Streamlit widget: align it with the content column and make it quiet */
.st-key-memory [data-testid="stHorizontalBlock"]{gap:48px!important;border-top:1px solid var(--rule);padding-top:40px}
.st-key-memory [data-testid="stColumn"]:first-child{flex:0 0 200px!important;width:200px!important;min-width:200px!important}
.st-key-memory [data-testid="stColumn"]:last-child{flex:1 1 0!important;min-width:0!important}
.st-key-memory input{background:var(--wash)!important;border:1px solid var(--rule)!important;border-radius:6px!important;
  font-family:var(--sans)!important;font-size:15px!important;color:var(--ink)!important}
.st-key-memory input:focus{border-color:var(--teal)!important;box-shadow:0 0 0 3px var(--teal-wash)!important}
.st-key-memory label{display:none}

@media (max-width:900px){
  .mast,.sec{grid-template-columns:minmax(0,1fr);gap:18px}
  .mast p{grid-column:1}
  .rail p{max-width:48ch}
  .big{font-size:68px}
  .rrow,.axis{grid-template-columns:minmax(0,1fr);gap:4px}
  .cw{left:0}
  .rname{padding-left:12px}
  .bars{gap:16px}
  .brow{row-gap:4px}
  .axis>div:first-child{display:none}
  .lhead{display:none}
  .lrow{grid-template-columns:24px minmax(0,1fr);gap:10px}
  .lrow .plot{grid-column:1/-1;margin:8px 0;min-height:36px}
  .lrow .nums{grid-column:1/-1;text-align:left}
  .brow,.gh{grid-template-columns:minmax(0,1fr) 56px}
  .brow .tr{grid-column:1/-1;order:3}
  .gh>div:first-child,.gh>div:last-child{display:none}
  .st-key-memory [data-testid="stHorizontalBlock"]{flex-direction:column;gap:12px!important}
  .st-key-memory [data-testid="stColumn"]{flex:1 1 auto!important;width:100%!important;min-width:0!important}
}
@media (prefers-reduced-motion:no-preference){
  .lrow{transition:background .12s}
}
"""