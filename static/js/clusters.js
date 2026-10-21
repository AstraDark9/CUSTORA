/* Customer clusters page. Reads window.CLUSTERS (set in clusters.html). */
(function () {
  'use strict';
  var C = window.CLUSTERS;
  var $ = function (id) { return document.getElementById(id); };
  var canvas = $('clusterMap'), tip = $('clusterTip'), legend = $('clusterLegend');
  var S = { data: null, pts: [], clamped: 0, hover: -1, active: -1, w: 0, h: 0, dpr: 1, x0: 0, x1: 1, y0: 0, y1: 1 };
  var PAD = { l: 14, r: 14, t: 14, b: 30 };
  var TONES = ['--accent', '--risk', '--watch', '--critical', '--healthy', '--ink-soft'];

  // Everything below can trace back to an uploaded file, so it is untrusted text.
  function esc(v) {
    if (v === null || v === undefined) return '';
    return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function group(n) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function money(n) {
    var c = C.currency, a = Math.abs(n);
    if (c === '\u20b9') {
      if (a >= 1e7) return c + (a / 1e7).toFixed(2) + 'Cr';
      if (a >= 1e5) return c + (a / 1e5).toFixed(1) + 'L';
    } else {
      if (a >= 1e6) return c + (a / 1e6).toFixed(2) + 'M';
      if (a >= 1e4) return c + (a / 1e3).toFixed(1) + 'K';
    }
    return c + group(a);
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function color(i) { return i < TONES.length ? cssVar(TONES[i]) : 'hsl(' + ((i * 57) % 360) + ',55%,55%)'; }
  function setNote(shortText, fullText) {
    $('clusterNoteShort').textContent = shortText;
    $('clusterNoteFull').textContent = fullText;
  }
  function percentile(sorted, p) { return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))]; }

  function xOf(v) { return PAD.l + (v - S.x0) / (S.x1 - S.x0) * (S.w - PAD.l - PAD.r); }
  function yOf(v) { return S.h - PAD.b - (v - S.y0) / (S.y1 - S.y0) * (S.h - PAD.t - PAD.b); }

  function layout() {
    if (!S.data) return;
    var r = canvas.getBoundingClientRect();
    S.dpr = window.devicePixelRatio || 1; S.w = r.width; S.h = r.height;
    canvas.width = Math.round(r.width * S.dpr); canvas.height = Math.round(r.height * S.dpr);
    S.clamped = 0;
    S.pts = S.data.x.map(function (x, i) {
      var y = S.data.y[i], cx = Math.min(S.x1, Math.max(S.x0, x)), cy = Math.min(S.y1, Math.max(S.y0, y));
      if (cx !== x || cy !== y) S.clamped++;
      return [xOf(cx), yOf(cy)];
    });
    draw();
  }

  function draw() {
    if (!S.data) return;
    var ctx = canvas.getContext('2d');
    ctx.setTransform(S.dpr, 0, 0, S.dpr, 0, 0);
    ctx.clearRect(0, 0, S.w, S.h);
    var faint = cssVar('--ink-faint'), line = cssVar('--line-strong');
    var plotL = PAD.l, plotR = S.w - PAD.r, plotT = PAD.t, plotB = S.h - PAD.b;
    ctx.strokeStyle = line; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
    ctx.strokeRect(plotL, plotT, plotR - plotL, plotB - plotT);
    ctx.globalAlpha = 1;
    ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = faint; ctx.textAlign = 'center';
    ctx.fillText('Buying pattern, main direction (PC1) \u2192', (plotL + plotR) / 2, S.h - 9);
    ctx.save(); ctx.translate(plotL + 11, (plotT + plotB) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('Second direction (PC2) \u2192', 0, 0); ctx.restore();

    var colors = S.data.segments.map(function (_, i) { return color(i); });
    var any = S.active >= 0;
    for (var i = 0; i < S.pts.length; i++) {
      var c = S.data.c[i], on = !any || c === S.active;
      ctx.globalAlpha = on ? 0.75 : 0.08;
      ctx.fillStyle = colors[c];
      ctx.beginPath(); ctx.arc(S.pts[i][0], S.pts[i][1], 3, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (S.hover >= 0) {
      ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(S.pts[S.hover][0], S.pts[S.hover][1], 6.5, 0, 6.2832); ctx.stroke();
    }
  }

  function nearest(mx, my) {
    var best = -1, bd = 12 * 12;
    for (var i = 0; i < S.pts.length; i++) {
      if (S.active >= 0 && S.data.c[i] !== S.active) continue;
      var dx = S.pts[i][0] - mx, dy = S.pts[i][1] - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  canvas.addEventListener('mousemove', function (e) {
    if (!S.data) return;
    var r = canvas.getBoundingClientRect(), i = nearest(e.clientX - r.left, e.clientY - r.top);
    if (i !== S.hover) { S.hover = i; draw(); }
    canvas.style.cursor = i >= 0 ? 'pointer' : 'default';
    if (i < 0) { tip.style.display = 'none'; return; }
    tip.innerHTML = '<b>' + esc(S.data.id[i]) + '</b><br>' + esc(S.data.segments[S.data.c[i]].name) + '<br>' + Math.round(S.data.p[i] * 100) + '% chance of leaving';
    tip.style.display = 'block';
    var tx = S.pts[i][0] + 14, ty = S.pts[i][1] - 10;
    if (tx + 220 > S.w) tx = S.pts[i][0] - 230;
    tip.style.left = Math.max(4, tx) + 'px'; tip.style.top = Math.max(4, ty) + 'px';
  });
  canvas.addEventListener('mouseleave', function () { S.hover = -1; tip.style.display = 'none'; draw(); });
  canvas.addEventListener('click', function (e) {
    if (!S.data) return;
    var r = canvas.getBoundingClientRect(), i = nearest(e.clientX - r.left, e.clientY - r.top);
    if (i >= 0) window.location.href = '/customer/' + encodeURIComponent(S.data.id[i]);
  });
  window.addEventListener('resize', layout);
  new MutationObserver(function () { renderLegend(); draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  function renderLegend() {
    if (!S.data) return;
    legend.innerHTML = S.data.segments.map(function (seg, i) {
      return '<button type="button" class="cluster-chip" data-seg="' + i + '" aria-pressed="' + (S.active === i) + '">' +
        '<span class="cluster-dot" style="background:' + color(i) + '"></span>' + esc(seg.name) +
        '<span class="cluster-n">' + group(seg.customers) + '</span></button>';
    }).join('');
  }
  legend.addEventListener('click', function (e) {
    var b = e.target.closest('[data-seg]'); if (!b) return;
    var i = parseInt(b.dataset.seg, 10);
    S.active = S.active === i ? -1 : i;
    renderLegend(); draw();
  });

  function renderTable(d) {
    $('clusterTable').querySelector('tbody').innerHTML = d.segments.map(function (seg, i) {
      return '<tr><td><span class="cluster-dot" style="background:' + color(i) + '"></span><span class="seg-name" style="display:inline">' + esc(seg.name) + '</span></td>' +
        '<td class="num">' + group(seg.customers) + '</td>' +
        '<td class="num">' + (seg.recency == null ? '\u2013' : group(seg.recency)) + '</td>' +
        '<td class="num">' + (seg.frequency == null ? '\u2013' : seg.frequency) + '</td>' +
        '<td class="num">' + (seg.monetary == null ? '\u2013' : money(seg.monetary)) + '</td>' +
        '<td class="num">' + Math.round(seg.avg_risk) + '%</td></tr>';
    }).join('');
  }

  fetch(C.dataUrl).then(function (r) { if (!r.ok) throw 0; return r.json(); }).then(function (d) {
    if (!d.available || !d.x.length) {
      setNote('The cluster map is not available.', 'The cluster map is not available for this analysis. Run the analysis again to create it.');
      return;
    }
    S.data = d;
    // Frame the main crowd; a handful of extreme customers would otherwise squash everyone else into a corner.
    var xs = d.x.slice().sort(function (a, b) { return a - b; }), ys = d.y.slice().sort(function (a, b) { return a - b; });
    var padx = (percentile(xs, 0.99) - percentile(xs, 0.01)) * 0.06 || 1, pady = (percentile(ys, 0.99) - percentile(ys, 0.01)) * 0.06 || 1;
    S.x0 = percentile(xs, 0.01) - padx; S.x1 = percentile(xs, 0.99) + padx;
    S.y0 = percentile(ys, 0.01) - pady; S.y1 = percentile(ys, 0.99) + pady;
    renderLegend(); renderTable(d); layout();
    var base = d.plotted < d.n ? 'Showing ' + group(d.plotted) + ' of ' + group(d.n) + ' customers.' : group(d.n) + ' customers.';
    var edge = S.clamped ? ' ' + group(S.clamped) + ' unusual customers sit far outside this view and are drawn on its edge.' : '';
    setNote(base, base + ' Customers who buy alike sit close together. The two axes are summaries of buying behaviour, so their numbers have no units.' + edge);
  }).catch(function () {
    setNote('Could not load the cluster data.', 'Could not load the cluster data. Refresh the page to retry.');
  });
})();
