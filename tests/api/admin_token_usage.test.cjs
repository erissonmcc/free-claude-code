const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor() { this.children = []; this.textContent = ""; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; this.textContent = ""; }
  text() { return this.textContent + this.children.map((child) => child.text()).join(" "); }
}

function dashboard() {
  const source = fs.readFileSync("src/free_claude_code/api/admin_static/admin.js", "utf8");
  const root = new Element();
  const context = vm.createContext({ document: { createElement: () => new Element() }, byId: () => root });
  vm.runInContext(source.slice(source.indexOf("function formatUsage("), source.indexOf("function renderUsageChart(")), context);
  return { context, root };
}

test("formatted usage ratios remain readable instead of becoming NaN", () => {
  const { context } = dashboard();
  const metric = context.usageMetric("Calls reporting cache", "1 / 2");
  assert.match(metric.text(), /1 \/ 2/);
  assert.doesNotMatch(metric.text(), /NaN/);
});

test("Vertex baseline presents unavailable costs and negative savings explicitly", () => {
  const { context, root } = dashboard();
  context.renderGoogleCacheBaseline([
    { provider: "vertex", model: "m", calls: 100, collected_days: 7, required_days: 7,
      required_calls: 100, eligible_calls: 100, status: "not_economical", cache_hit_rate: 1,
      prefix_reuse_rate: .99, estimated_net_savings_usd: -.01,
      estimated_cache_storage_usd: .0001667 },
  ]);
  const text = root.text();
  assert.match(text, /vertex \/ m/);
  assert.match(text, /100\.0%/);
  assert.match(text, /-\$0\.01/);
  assert.match(text, /Unavailable/);
  assert.match(text, /not economical/i);
  assert.doesNotMatch(text, /NaN|undefined/);
});
