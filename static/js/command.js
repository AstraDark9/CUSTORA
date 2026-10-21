/* Retention command center. Reads window.CMD (set by the page). Used by the dashboard (key figures + action queue),
   the Risk × value page (matrix) and the Outreach planner page; every block below only runs when its elements exist. */
(function () {
  'use strict';
  var CMD = window.CMD;
  var $ = function (id) { return document.getElementById(id); };

  // ---------- helpers ----------
  // Everything below can trace back to an uploaded file, so it is untrusted text.
  function esc(v) {
    if (v === null || v === undefined) return '';
    return String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function group(n) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function money(n) {
    var c = CMD.currency, a = Math.abs(n), s = n < 0 ? '-' : '';
    if (c === '\u20b9') {
      if (a >= 1e7) return s + c + (a / 1e7).toFixed(2) + 'Cr';
      if (a >= 1e5) return s + c + (a / 1e5).toFixed(1) + 'L';
    } else {
      if (a >= 1e6) return s + c + (a / 1e6).toFixed(2) + 'M';
      if (a >= 1e4) return s + c + (a / 1e3).toFixed(1) + 'K';
    }
    return s + c + group(a);
  }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  // ---------- state shared by the matrix, the queue and the planner ----------
  var state = { quad: '__all__', stateName: '__all__', sort: 'loss', q: '', offset: 0, total: 0, token: 0 };
  var quadLabels = {};
  Object.keys(CMD.quadMeta || {}).forEach(function (k) { quadLabels[k] = CMD.quadMeta[k].title; });

  function toast(msg) {
    var t = $('toast'); if (!t) return;
    t.textContent = msg; t.classList.add('show');
    clearTimeout(toast._t); toast._t = setTimeout(function () { t.classList.remove('show'); }, 2600);
  }
  // Two-part text ("short until you hover"): set both parts, if the element exists.
  function setReveal(prefix, shortText, fullText) {
    var a = $(prefix + 'Short'), b = $(prefix + 'Full');
    if (a) a.textContent = shortText;
    if (b) b.textContent = fullText;
  }

  // =====================================================================
  // RISK x VALUE MATRIX
  // =====================================================================
  var M = { data: null, canvas: $('matrix'), tip: $('matrixTip'), pts: [], hover: -1, w: 0, h: 0, dpr: 1 };
  var hasQueue = !!$('rows');
  var PAD = { l: 62, r: 14, t: 14, b: 38 };
  var QUAD_TONE = ['--critical', '--healthy', '--watch', '--ink-faint'];

  function sizeCanvas() {
    var r = M.canvas.getBoundingClientRect();
    M.dpr = window.devicePixelRatio || 1;
    M.w = r.width; M.h = r.height;
    M.canvas.width = Math.round(r.width * M.dpr);
    M.canvas.height = Math.round(r.height * M.dpr);
  }
  function xOf(p) { return PAD.l + p * (M.w - PAD.l - PAD.r); }
  function yOf(v) {
    var lo = M.vmin, hi = M.vmax;
    var t = (Math.log(Math.max(v, lo)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
    return M.h - PAD.b - t * (M.h - PAD.t - PAD.b);
  }

  function drawMatrix() {
    if (!M.data || !M.canvas) return;
    var ctx = M.canvas.getContext('2d');
    ctx.setTransform(M.dpr, 0, 0, M.dpr, 0, 0);
    ctx.clearRect(0, 0, M.w, M.h);
    var ink = cssVar('--ink-soft'), line = cssVar('--line-strong'), faint = cssVar('--ink-faint');
    var tones = QUAD_TONE.map(function (n) { return cssVar(n); });
    var xr = xOf(M.data.risk_cut), yv = yOf(M.data.value_cut);
    var plotL = PAD.l, plotR = M.w - PAD.r, plotT = PAD.t, plotB = M.h - PAD.b;

    // quadrant tints
    function tint(x, y, w, h, color, a) { ctx.globalAlpha = a; ctx.fillStyle = color; ctx.fillRect(x, y, w, h); ctx.globalAlpha = 1; }
    var active = M.data.quadrants.indexOf(state.quad);
    tint(xr, plotT, plotR - xr, yv - plotT, tones[0], active === 0 ? 0.16 : 0.07);
    tint(plotL, plotT, xr - plotL, yv - plotT, tones[1], active === 1 ? 0.16 : 0.05);
    tint(xr, yv, plotR - xr, plotB - yv, tones[2], active === 2 ? 0.16 : 0.05);

    // grid + axes
    ctx.font = '11px Inter, sans-serif'; ctx.fillStyle = faint; ctx.strokeStyle = line; ctx.lineWidth = 1;
    ctx.textAlign = 'center';
    for (var i = 0; i <= 5; i++) {
      var p = i / 5, x = xOf(p);
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(x, plotT); ctx.lineTo(x, plotB); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(Math.round(p * 100) + '%', x, plotB + 15);
    }
    ctx.textAlign = 'right';
    var decades = [];
    for (var d = Math.pow(10, Math.ceil(Math.log10(M.vmin))); d <= M.vmax; d *= 10) decades.push(d);
    decades.forEach(function (v) {
      var y = yOf(v);
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(plotL, y); ctx.lineTo(plotR, y); ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillText(money(v), plotL - 6, y + 4);
    });
    ctx.textAlign = 'center'; ctx.fillStyle = ink;
    ctx.fillText('Chance the customer leaves \u2192', (plotL + plotR) / 2, M.h - 6);
    ctx.save(); ctx.translate(12, (plotT + plotB) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('Revenue at stake (log) \u2192', 0, 0); ctx.restore();

    // dividing lines
    ctx.strokeStyle = faint; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(xr, plotT); ctx.lineTo(xr, plotB); ctx.moveTo(plotL, yv); ctx.lineTo(plotR, yv); ctx.stroke(); ctx.setLineDash([]);

    // dots: dimmed unless they are in the selected group; biggest expected loss on top
    var order = M.order;
    for (var k = 0; k < order.length; k++) {
      var j = order[k], q = M.data.q[j], on = (state.quad === '__all__' || M.data.quadrants[q] === state.quad);
      var r = 2.2 + 3.4 * Math.sqrt(M.data.l[j] / M.maxLoss);
      ctx.globalAlpha = on ? 0.78 : 0.1;
      ctx.fillStyle = tones[q];
      ctx.beginPath(); ctx.arc(M.pts[j][0], M.pts[j][1], r, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (M.hover >= 0) {
      var h = M.pts[M.hover];
      ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(h[0], h[1], 7, 0, 6.2832); ctx.stroke();
    }
    // corner labels
    ctx.font = '700 11px Inter, sans-serif'; ctx.textAlign = 'right';
    ctx.fillStyle = tones[0]; ctx.fillText('SAVE NOW', plotR - 8, plotT + 15);
    ctx.textAlign = 'left'; ctx.fillStyle = tones[1]; ctx.fillText('PROTECT', plotL + 8, plotT + 15);
    ctx.textAlign = 'right'; ctx.fillStyle = tones[2]; ctx.fillText('WIN BACK CHEAPLY', plotR - 8, plotB - 8);
  }

  function layoutMatrix() {
    if (!M.data || !M.canvas) return;
    sizeCanvas();
    M.pts = M.data.p.map(function (p, i) { return [xOf(p), yOf(M.data.v[i])]; });
    drawMatrix();
  }

  function loadMatrix() {
    fetch(CMD.matrixUrl).then(function (r) { if (!r.ok) throw 0; return r.json(); }).then(function (d) {
      M.data = d;
      var vs = d.v.filter(function (v) { return v > 0; });
      M.vmin = Math.max(1, Math.min.apply(null, vs.concat([d.value_cut])) * 0.8);
      M.vmax = Math.max.apply(null, vs.concat([d.value_cut])) * 1.25;
      M.maxLoss = Math.max.apply(null, d.l.concat([1]));
      M.order = d.l.map(function (_, i) { return i; }).sort(function (a, b) { return d.l[a] - d.l[b]; });
      if (d.plotted < d.n) {
        var shown = 'Showing ' + group(d.plotted) + ' of ' + group(d.n) + ' customers.';
        setReveal('matrixNote', shown, shown + ' All the largest expected losses plus a sample of the rest. Numbers elsewhere use everyone.');
      } else {
        setReveal('matrixNote', 'Dot size = expected loss. Dashed lines split the groups.',
          'Dot size = expected loss. Dashed lines: the "likely to leave" line (' + Math.round(d.risk_cut * 100) + '% chance) and the "high value" line (top third by revenue at stake, ' + money(d.value_cut) + ').');
      }
      if ($('capacity')) initPlanner(d);
      layoutMatrix();
    }).catch(function () { setReveal('matrixNote', 'Could not load the chart data. Refresh the page to retry.', 'Could not load the chart data. Refresh the page to retry.'); });
  }

  function nearest(mx, my) {
    var best = -1, bd = 14 * 14;
    for (var i = 0; i < M.pts.length; i++) {
      var q = M.data.quadrants[M.data.q[i]];
      if (state.quad !== '__all__' && q !== state.quad) continue;
      var dx = M.pts[i][0] - mx, dy = M.pts[i][1] - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  if (M.canvas) {
    M.canvas.addEventListener('mousemove', function (e) {
      if (!M.data) return;
      var r = M.canvas.getBoundingClientRect(), i = nearest(e.clientX - r.left, e.clientY - r.top);
      if (i !== M.hover) { M.hover = i; drawMatrix(); }
      if (i < 0) { M.tip.style.display = 'none'; return; }
      M.tip.innerHTML = '<b>' + esc(M.data.id[i]) + '</b><br>' + Math.round(M.data.p[i] * 100) + '% chance of leaving<br>' +
        money(M.data.v[i]) + ' at stake \u00b7 ' + money(M.data.l[i]) + ' expected loss';
      M.tip.style.display = 'block';
      var tx = M.pts[i][0] + 14, ty = M.pts[i][1] - 10;
      if (tx + 220 > M.w) tx = M.pts[i][0] - 230;
      M.tip.style.left = Math.max(4, tx) + 'px'; M.tip.style.top = Math.max(4, ty) + 'px';
    });
    M.canvas.addEventListener('mouseleave', function () { M.hover = -1; M.tip.style.display = 'none'; drawMatrix(); });
    M.canvas.addEventListener('click', function (e) {
      if (!M.data) return;
      var r = M.canvas.getBoundingClientRect(), i = nearest(e.clientX - r.left, e.clientY - r.top);
      if (i >= 0) openDrawer(M.data.id[i]);
    });
    window.addEventListener('resize', layoutMatrix);
    new MutationObserver(drawMatrix).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  // quadrant buttons
  document.querySelectorAll('.quad').forEach(function (b) {
    b.addEventListener('click', function () { setQuad(state.quad === b.dataset.quad ? '__all__' : b.dataset.quad); });
  });
  if ($('clearQuad')) $('clearQuad').addEventListener('click', function () { setQuad('__all__'); });
  function setQuad(key) {
    state.quad = key;
    document.querySelectorAll('.quad').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.quad === key ? 'true' : 'false'); });
    var btn = document.querySelector('.quad[data-quad="' + key + '"]');
    if ($('quadPlay')) $('quadPlay').innerHTML = btn ? '<strong>' + esc(quadLabels[key]) + ':</strong> ' + esc(btn.dataset.play) : 'Select a group to see what to do with it.';
    // Risk x value page: offer the matching list on the command center.
    var link = $('quadQueueLink');
    if (link) {
      link.hidden = key === '__all__';
      if (key !== '__all__') link.href = CMD.dashboardUrl + '?quadrant=' + encodeURIComponent(key) + '#priority-queue';
    }
    // Command center: the group shows as a removable filter on the queue.
    if ($('clearQuad')) {
      $('clearQuad').classList.toggle('on', key !== '__all__');
      $('clearQuadText').textContent = key === '__all__' ? '' : quadLabels[key];
    }
    drawMatrix();
    if (hasQueue) {
      loadQueue(true);
      if (key !== '__all__') $('priority-queue').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // =====================================================================
  // OUTREACH PLANNER (ranked by expected loss; everything recomputed live)
  // =====================================================================
  var P = { loss: [], prob: [], cum: [], cumP: [], total: 0 };
  function initPlanner(d) {
    var idx = d.l.map(function (_, i) { return i; }).sort(function (a, b) { return d.l[b] - d.l[a]; });
    var c = 0, cp = 0;
    P.loss = idx.map(function (i) { return d.l[i]; });
    P.prob = idx.map(function (i) { return d.p[i]; });
    P.cum = P.loss.map(function (v) { c += v; return c; });
    P.cumP = P.prob.map(function (v) { cp += v; return cp; });
    P.total = c;
    var slider = $('capacity');
    slider.max = String(P.loss.length);
    slider.step = String(Math.max(1, Math.round(P.loss.length / 100)));
    slider.value = String(Math.min(P.loss.length, Math.max(10, Math.round(P.loss.length * 0.1))));
    renderPlanner();
  }
  function gainAt(pct) {
    var g = CMD.gain; if (!g || !g.length) return null;
    var best = g[0];
    g.forEach(function (row) { if (Math.abs(row.ContactedPct - pct) < Math.abs(best.ContactedPct - pct)) best = row; });
    return best;
  }
  function renderPlanner() {
    if (!P.loss.length) return;
    var n = parseInt($('capacity').value, 10), rate = parseInt($('saveRate').value, 10) / 100;
    var cost = Math.max(0, parseFloat($('cost').value) || 0);
    var covered = P.cum[n - 1], leavers = P.cumP[n - 1];
    var saved = covered * rate, net = saved - n * cost;
    $('capacityOut').textContent = group(n) + ' (' + Math.round(n / P.loss.length * 100) + '%)';
    $('saveRateOut').textContent = Math.round(rate * 100) + '%';
    $('rLeavers').textContent = '\u2248 ' + group(leavers);
    $('rCovered').textContent = money(covered) + ' (' + Math.round(covered / P.total * 100) + '%)';
    $('rSaved').textContent = money(saved);
    var netEl = $('rNet'); netEl.textContent = money(net); netEl.className = 'r-val ' + (net >= 0 ? 'good' : 'bad');
    var last = P.loss[n - 1], pct = n / P.loss.length * 100, g = gainAt(pct);
    var lead = 'The ' + group(n) + 'th customer on this list is worth about ' + money(last * rate) + ' to you in expectation, against ' + money(cost) + ' to contact.';
    var check = g ? ' Reality check: on the held-out period, contacting the riskiest ' + Math.round(g.ContactedPct) + '% of customers reached ' + Math.round(g.ChurnersReachedPct) + '% of those who really left (' + Math.round(g.HitRatePct) + '% of contacts were genuine leavers).' : '';
    setReveal('rEvidence', 'Customer #' + group(n) + ' is worth about ' + money(last * rate) + ' to you, against ' + money(cost) + ' to contact.', lead + check);
    $('exportLink').href = CMD.exportUrl + '?sort=loss&limit=' + n;
  }
  if ($('capacity')) {
    ['capacity', 'saveRate', 'cost'].forEach(function (id) { $(id).addEventListener('input', renderPlanner); });
    $('useBest').addEventListener('click', function () {
      var rate = parseInt($('saveRate').value, 10) / 100, cost = Math.max(0, parseFloat($('cost').value) || 0), best = 1;
      for (var i = 0; i < P.loss.length; i++) { if (P.loss[i] * rate >= cost) best = i + 1; else break; }
      $('capacity').value = String(best); renderPlanner();
    });
  }

  // =====================================================================
  // ACTION QUEUE
  // =====================================================================
  var rowsEl = $('rows'), mobileEl = $('mobileCards'), noRes = $('noResults'), moreBtn = $('moreBtn');

  function riskClass(p) { return p >= 0.55 ? 'risk' : (p >= 0.35 ? 'watch' : 'healthy'); }
  function rowHtml(c) {
    var id = esc(String(c.CustomerID)), pct = Math.round((c.PredictedChurnProbability || 0) * 100);
    return '<tr class="cust-row" tabindex="0" role="button" data-customer="' + id + '">' +
      '<td><div class="cust-id">' + id + '</div><div class="cust-state">' + esc(c.CustomerState) + ' \u00b7 ' + esc(c.SegmentName) + '</div></td>' +
      '<td><span class="risk-pct ' + riskClass(c.PredictedChurnProbability) + '">' + pct + '%</span></td>' +
      '<td class="num">' + money(c.RevenueAtStake || 0) + '</td>' +
      '<td class="num"><b>' + money(c.ExpectedLoss || 0) + '</b></td>' +
      '<td><div class="act-text">' + esc(c.Recommendation || '') + '</div></td>' +
      '<td class="row-actions"><button type="button" class="row-reached" data-reach="' + id + '" data-priority="' + esc(c.BusinessPriority || '') + '" title="Mark as reached" aria-label="Mark customer ' + id + ' as reached">Reached \u2713</button>' +
      '<span class="row-arrow"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg></span></td></tr>';
  }
  function cardHtml(c) {
    var id = esc(String(c.CustomerID)), pct = Math.round((c.PredictedChurnProbability || 0) * 100);
    return '<button type="button" class="mobile-cust-card" data-customer="' + id + '">' +
      '<div class="mc-top"><span class="cust-id">' + id + '</span><span class="risk-pct ' + riskClass(c.PredictedChurnProbability) + '">' + pct + '% risk</span></div>' +
      '<div class="mc-line"><b>' + money(c.ExpectedLoss || 0) + '</b> expected loss \u00b7 ' + money(c.RevenueAtStake || 0) + ' at stake</div>' +
      '<div class="mc-line">' + esc(c.Recommendation || '') + '</div></button>';
  }

  function loadQueue(reset) {
    if (!hasQueue) return;
    if (reset) { state.offset = 0; rowsEl.innerHTML = ''; mobileEl.innerHTML = ''; }
    var token = ++state.token;
    var params = new URLSearchParams({ state: state.stateName, quadrant: state.quad, q: state.q, sort: state.sort, limit: 50, offset: state.offset });
    fetch(CMD.listUrl + '?' + params.toString()).then(function (r) { if (!r.ok) throw 0; return r.json(); }).then(function (d) {
      if (token !== state.token) return;
      rowsEl.insertAdjacentHTML('beforeend', d.rows.map(rowHtml).join(''));
      mobileEl.insertAdjacentHTML('beforeend', d.rows.map(cardHtml).join(''));
      state.offset += d.rows.length; state.total = d.total;
      noRes.style.display = d.total === 0 ? 'block' : 'none';
      var scope = state.quad !== '__all__' ? ' in \u201c' + quadLabels[state.quad] + '\u201d' : '';
      $('pagerText').textContent = 'Showing ' + group(state.offset) + ' of ' + group(d.total) + scope;
      moreBtn.style.display = state.offset < d.total ? '' : 'none';
    }).catch(function () {
      noRes.textContent = 'Could not load customers. Please refresh the page.'; noRes.style.display = 'block';
    });
  }
  if (hasQueue) {
    moreBtn.addEventListener('click', function () { loadQueue(false); });
    var timer = null;
    $('customerSearch').addEventListener('input', function (e) {
      clearTimeout(timer); timer = setTimeout(function () { state.q = e.target.value.trim(); loadQueue(true); }, 250);
    });
    $('stateSelect').addEventListener('change', function (e) { state.stateName = e.target.value; loadQueue(true); });
    $('sortSelect').addEventListener('change', function (e) { state.sort = e.target.value; loadQueue(true); });
  }
  // delegated: rows are created after load
  document.addEventListener('click', function (e) {
    var reach = e.target.closest('[data-reach]');
    if (reach) { markReached(reach.dataset.reach, { BusinessPriority: reach.dataset.priority }, reach); return; }
    var el = e.target.closest('[data-customer]'); if (el) openDrawer(el.dataset.customer);
  });
  document.addEventListener('keydown', function (e) {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.matches && e.target.matches('tr.cust-row')) { e.preventDefault(); openDrawer(e.target.dataset.customer); }
  });
  // sidebar "jump to a state"
  document.querySelectorAll('[data-jump-state]').forEach(function (b) {
    b.addEventListener('click', function () {
      var close = $('sidebarClose'); if (close && getComputedStyle(close).display !== 'none') close.click();
      $('stateSelect').value = b.dataset.jumpState; state.stateName = b.dataset.jumpState;
      loadQueue(true); $('priority-queue').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  // =====================================================================
  // CUSTOMER DRAWER
  // =====================================================================
  var drawer = $('customerDrawer'), overlay = $('drawerOverlay'), body = $('drawerBody'), reqToken = 0, lastFocus = null;
  var drawerCustomer = null;
  var HEALTH = ['healthy', 'watch', 'critical'];
  var fallbackAction = {
    Urgent: 'Escalate to a retention specialist within 48 hours.', High: 'Schedule proactive retention outreach within 7 days.',
    Medium: 'Add to the next scheduled check-in cycle.', Low: 'No action needed; continue standard engagement.'
  };
  function skeleton() {
    return '<div class="skeleton" style="height:78px;margin-bottom:22px"></div><div class="skeleton" style="height:16px;width:40%;margin-bottom:14px"></div>' +
      '<div class="skeleton" style="height:44px;margin-bottom:10px"></div><div class="skeleton" style="height:44px;margin-bottom:10px"></div><div class="skeleton" style="height:90px"></div>';
  }
  function renderDrawer(data) {
    var c = data.customer, pr = fallbackAction[c.RecommendationPriority] ? c.RecommendationPriority : 'Medium';
    var pct = c.PredictedChurnProbability != null ? Math.round(c.PredictedChurnProbability * 100) : null;
    var tone = pct == null ? 'watch' : (pct >= 55 ? 'risk' : (pct >= 35 ? 'watch' : 'healthy'));
    var healthKey = c.RelationshipHealth === 'Healthy' ? 'healthy' : (c.RelationshipHealth === 'At Risk' ? 'watch' : 'critical');
    $('drawerCustId').textContent = c.CustomerID || '\u2014';
    $('drawerPriorityBadge').innerHTML = '<span class="badge ' + esc(pr) + '"><span class="dotb"></span>' + esc(pr) + ' priority</span>';
    $('drawerProfileLink').href = data.profile_url || '#';
    drawerCustomer = c;
    $('drawerReachedBtn').disabled = false;
    var reasons = data.churn_reasons || [], maxMag = Math.max.apply(null, reasons.map(function (r) { return r.magnitude || 0; }).concat([0.0001]));
    var rh = reasons.length ? reasons.map(function (r) {
      var dir = r.direction === 'risk' ? 'risk' : 'reassurance', w = r.magnitude != null ? Math.max(8, Math.round(r.magnitude / maxMag * 100)) : 40;
      return '<div class="drawer-reason"><div class="reason-arrow ' + dir + '">' + (dir === 'risk' ? '\u25b2' : '\u25bc') + '</div><div class="drawer-reason-body">' +
        '<div class="drawer-reason-label">' + esc(r.label) + '</div><div class="drawer-reason-text">' + esc(r.text) + '</div>' +
        '<div class="contrib-bar-track"><div class="contrib-bar-fill ' + dir + '" style="width:' + w + '%"></div></div></div></div>';
    }).join('') : '<div class="drawer-empty">No churn explanation available for this customer.</div>';
    var ev = c.RecommendationEvidence ? c.RecommendationEvidence.split(' | ').map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('')
      : '<li>' + (c.Recency != null ? Math.round(c.Recency) + ' days since last purchase' : 'Recency not available') + '</li>';
    body.innerHTML =
      '<div class="drawer-stats">' +
      '<div class="drawer-stat"><div class="drawer-stat-label">Churn risk</div><div class="drawer-stat-value" style="color:var(--' + tone + ')">' + (pct != null ? pct + '%' : 'N/A') + '</div></div>' +
      '<div class="drawer-stat"><div class="drawer-stat-label">At stake</div><div class="drawer-stat-value">' + (c.RevenueAtStake != null ? money(c.RevenueAtStake) : 'N/A') + '</div></div>' +
      '<div class="drawer-stat"><div class="drawer-stat-label">Expected loss</div><div class="drawer-stat-value" style="color:var(--risk)">' + (c.ExpectedLoss != null ? money(c.ExpectedLoss) : 'N/A') + '</div></div></div>' +
      '<div class="drawer-section"><div class="drawer-section-title">Why this customer may leave</div>' + rh + '</div>' +
      '<div class="drawer-section"><div class="drawer-section-title">Recommended action</div><div class="rec-panel">' +
      '<div class="rec-panel-title">' + (esc(c.Recommendation) || 'No recommendation available') + '</div>' +
      '<div class="rec-panel-body">' + (esc(c.RecommendationAction) || fallbackAction[pr]) + '</div>' +
      '<ul class="rec-panel-evidence">' + ev + '<li>Relationship health: ' + esc(c.RelationshipHealth || 'N/A') + ' \u00b7 ' + esc(c.SegmentName || 'N/A') + '</li></ul></div></div>';
  }
  function focusables() {
    return Array.prototype.slice.call(drawer.querySelectorAll('a[href], button:not([disabled])')).filter(function (el) { return el.offsetParent !== null; });
  }
  function openDrawer(id) {
    if (!drawer) return;
    lastFocus = document.activeElement;
    drawerCustomer = null; $('drawerReachedBtn').disabled = true;
    drawer.classList.add('open'); overlay.classList.add('open'); drawer.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    $('drawerCustId').textContent = id; $('drawerPriorityBadge').innerHTML = ''; body.innerHTML = skeleton();
    $('drawerCloseBtn').focus();
    var t = ++reqToken;
    fetch('/customer/' + encodeURIComponent(id) + '/panel.json').then(function (r) { if (!r.ok) throw 0; return r.json(); }).then(function (d) {
      if (t === reqToken) renderDrawer(d);
    }).catch(function () {
      if (t !== reqToken) return;
      body.innerHTML = '<div class="drawer-empty">Couldn\u2019t load this customer. <a href="/customer/' + encodeURIComponent(id) + '">Open the full profile instead</a>.</div>';
    });
  }
  function closeDrawer() {
    drawer.classList.remove('open'); overlay.classList.remove('open'); drawer.setAttribute('aria-hidden', 'true');
    document.body.style.overflow = '';
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  if (drawer) {
    overlay.addEventListener('click', closeDrawer);
    $('drawerCloseBtn').addEventListener('click', closeDrawer);
    $('drawerReachedBtn').addEventListener('click', function () {
      if (drawerCustomer) markReached(String(drawerCustomer.CustomerID), drawerCustomer, $('drawerReachedBtn'));
    });
  }
  document.addEventListener('keydown', function (e) {
    if (!drawer || !drawer.classList.contains('open')) return;
    if (e.key === 'Escape') { closeDrawer(); return; }
    if (e.key === 'Tab') {
      var f = focusables(); if (!f.length) return;
      if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
      else if (!e.shiftKey && document.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
    }
  });

  // =====================================================================
  // MARK AS REACHED (takes the customer off the to-do list; they are listed on the Reached page)
  // =====================================================================
  function bumpKpis(priority) {
    var el = $('kpiActNow'); if (!el) return;
    function minusOne(node) { node.textContent = group(Math.max(0, parseInt(node.textContent.replace(/,/g, ''), 10) - 1)); }
    if (priority === 'Critical Business Priority' || priority === 'High Business Priority') minusOne(el);
    if (priority === 'Critical Business Priority' && $('kpiCritical')) minusOne($('kpiCritical'));
  }
  function markReached(id, customer, btn) {
    if (btn) btn.disabled = true;
    fetch('/customers/' + encodeURIComponent(id) + '/reached', { method: 'POST', headers: { 'Accept': 'application/json' } })
      .then(function (r) { if (!r.ok) throw 0; return r.json(); })
      .then(function (d) {
        var badge = $('reachedBadge');
        if (badge) { badge.textContent = d.reached_count; badge.hidden = !d.reached_count; }
        toast('Customer ' + id + ' moved to Reached customers.');
        if (customer) bumpKpis(customer.BusinessPriority);
        if (drawer && drawer.classList.contains('open')) closeDrawer();
        if (hasQueue) loadQueue(true);
        // The matrix page's group counts are drawn by the server, so refresh the page to keep them right.
        if (M.canvas) setTimeout(function () { location.reload(); }, 900);
      })
      .catch(function () { if (btn) btn.disabled = false; toast('Could not save this. Please try again.'); });
  }

  // ---------- go ----------
  if (M.canvas || $('capacity')) loadMatrix();
  var startQuad = new URLSearchParams(location.search).get('quadrant');
  if (hasQueue && startQuad && quadLabels[startQuad]) setQuad(startQuad);
  else loadQueue(true);
})();
