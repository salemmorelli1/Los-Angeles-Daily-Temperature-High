/* Calendar, publication and snapshot rules shared by the page and Node tests. */
(function (root) {
  "use strict";
  const DAY = 86400000;
  const missing = value => value == null || ["", "nan", "none", "null"].includes(String(value).trim().toLowerCase());
  const number = value => {
    if (missing(value) || typeof value === "boolean") return null;
    const result = Number(value);
    return Number.isFinite(result) ? result : null;
  };
  function published(row) {
    return !!row && ["NORMAL", "CAUTION"].includes(row.publish_mode)
      && [row.forecast_finalized_at, row.governance_run_at].some(value => !missing(value));
  }
  function forecast(row, horizon) {
    return published(row) ? number(row[`forecast_h${horizon}`]) : null;
  }
  function isoDay(value) {
    if (missing(value)) return null;
    const day = String(value).slice(0, 10);
    const parsed = Date.parse(`${day}T00:00:00Z`);
    return /^\d{4}-\d{2}-\d{2}$/.test(day) && Number.isFinite(parsed)
      && new Date(parsed).toISOString().slice(0, 10) === day ? day : null;
  }
  function targetDate(row, horizon) {
    const feature = isoDay(row?.feature_date);
    if (!feature) return null;
    const expected = new Date(Date.parse(`${feature}T00:00:00Z`) + horizon * DAY).toISOString().slice(0, 10);
    const explicit = row?.[`target_date_h${horizon}`];
    return missing(explicit) ? expected : isoDay(explicit) === expected ? expected : null;
  }
  function targetStatus(row, horizon, now = new Date()) {
    const target = targetDate(row, horizon);
    if (!target) return "Target unavailable";
    const parts = Object.fromEntries(new Intl.DateTimeFormat("en-US", {
      timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit"
    }).formatToParts(now).map(part => [part.type, part.value]));
    const today = `${parts.year}-${parts.month}-${parts.day}`;
    return target < today ? "Past target · archived forecast"
      : target === today ? "Today's daily high" : "Upcoming daily high";
  }
  function historySeries(rows, horizon) {
    const points = rows.filter(published).map(row => ({
      row, target: targetDate(row, horizon)
    })).filter(point => point.target).sort((a, b) => a.target.localeCompare(b.target));
    const series = {x: [], forecast: [], realized: [], residual: [], customdata: []};
    let previous = null;
    const append = (x, f, real, custom) => {
      series.x.push(x); series.forecast.push(f); series.realized.push(real);
      series.residual.push(f !== null && real !== null ? real - f : null);
      series.customdata.push(custom);
    };
    for (const point of points) {
      const {row, target} = point;
      if (previous && (Date.parse(target) - Date.parse(previous.target) > DAY
          || row.training_protocol_version !== previous.row.training_protocol_version)) {
        append(new Date(Date.parse(previous.target) + DAY / 2).toISOString(), null, null, ["", "", ""]);
      }
      append(target, forecast(row, horizon), number(row[`realized_h${horizon}`]), [
        row.decision_date, row.feature_date, row.training_protocol_version || "legacy / unspecified"
      ]);
      previous = point;
    }
    return series;
  }
  async function loadSnapshot(base, specs, parse, fetcher = root.fetch, crypto = root.crypto) {
    const get = async path => {
      const response = await fetcher(`${base}/${path}?v=${Date.now()}`, {cache: "no-store"});
      if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
      return response;
    };
    const manifestText = await (await get("publication.json")).text();
    const manifest = JSON.parse(manifestText);
    if (manifest.schema_version !== "LA_PUBLICATION_V1" || !manifest.files || !crypto?.subtle) {
      throw new Error("Verified publication unavailable");
    }
    const entries = await Promise.all(specs.map(async ([key, path, type]) => {
      const record = manifest.files[path];
      if (!record || !/^[0-9a-f]{64}$/.test(record.sha256)) throw new Error(`Unlisted artifact: ${path}`);
      const bytes = await (await get(path)).arrayBuffer();
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
        .map(byte => byte.toString(16).padStart(2, "0")).join("");
      if (bytes.byteLength !== record.size_bytes || digest !== record.sha256) {
        throw new Error(`Snapshot integrity mismatch: ${path}. Refresh after deployment completes.`);
      }
      return [key, parse(new TextDecoder().decode(bytes), type)];
    }));
    if (await (await get("publication.json")).text() !== manifestText) {
      throw new Error("Publication changed during loading. Refresh to load one snapshot.");
    }
    const artifacts = Object.fromEntries(entries);
    const diagnostics = artifacts.diagnostics;
    if (diagnostics?.schema_version !== "LA_FORECAST_DIAGNOSTICS_V1"
        || diagnostics.integrity_status !== "PASS" || diagnostics.published !== true
        || diagnostics.decision_date !== manifest.decision_date) {
      throw new Error("Forecast diagnostics and publication disagree");
    }
    const rows = artifacts.predictionLog || [];
    const latest = [...rows].sort((a, b) => String(a.feature_date).localeCompare(String(b.feature_date))).at(-1);
    if (!published(latest) || latest.decision_date !== diagnostics.decision_date
        || latest.publish_mode !== diagnostics.publish_mode
        || latest.forecast_source !== diagnostics.forecast_source
        || artifacts.governance?.decision_date !== diagnostics.decision_date
        || artifacts.governance?.publish_mode !== diagnostics.publish_mode
        || artifacts.governance?.forecast_source !== diagnostics.forecast_source
        || [1, 3, 5].some(h => forecast(latest, h) === null
          || forecast(latest, h) !== diagnostics.horizons?.[`h${h}`]?.forecast_f
          || targetDate(latest, h) !== diagnostics.horizons?.[`h${h}`]?.target_date)) {
      throw new Error("Canonical forecast and publication disagree");
    }
    return {artifacts, manifest};
  }
  const api = Object.freeze({published, forecast, targetDate, targetStatus, historySeries, loadSnapshot});
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.LAContract = api;
})(typeof globalThis === "undefined" ? window : globalThis);
