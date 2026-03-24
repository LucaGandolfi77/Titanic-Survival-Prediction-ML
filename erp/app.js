// ═══════════════════════════════════════════════════════════════
// ERP System — Frontend Application
// ═══════════════════════════════════════════════════════════════

(function () {
  "use strict";

  // ─── Configuration ──────────────────────────────────────────
  const API = "http://localhost:8000";
  let token = localStorage.getItem("erp_token") || "";
  let currentUser = null;
  let revenueChart = null;
  let plChart = null;

  // ─── DOM refs ───────────────────────────────────────────────
  const $ = (s, p) => (p || document).querySelector(s);
  const $$ = (s, p) => [...(p || document).querySelectorAll(s)];
  const loginScreen = $("#login-screen");
  const appShell = $("#app-shell");
  const mainContent = $("#main-content");
  const loginForm = $("#login-form");
  const loginError = $("#login-error");
  const defaultPwWarn = $("#default-pw-warning");
  const sidebar = $("#sidebar");
  const sidebarToggle = $("#sidebar-toggle");
  const userInfo = $("#user-info");
  const logoutBtn = $("#logout-btn");
  const toastContainer = $("#toast-container");
  const modalOverlay = $("#modal-overlay");
  const modalTitle = $("#modal-title");
  const modalBody = $("#modal-body");
  const modalFooter = $("#modal-footer");
  const modalClose = $("#modal-close");
  const confirmOverlay = $("#confirm-overlay");
  const confirmTitle = $("#confirm-title");
  const confirmBody = $("#confirm-body");
  const confirmOk = $("#confirm-ok");
  const confirmCancel = $("#confirm-cancel");

  // ─── API Client ─────────────────────────────────────────────
  async function api(method, path, body) {
    const opts = {
      method,
      headers: { "Content-Type": "application/json" },
    };
    if (token) opts.headers["Authorization"] = "Bearer " + token;
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(API + path, opts);
    if (res.status === 401) {
      logout();
      throw new Error("Session expired");
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Request failed");
    }
    return res.json();
  }

  // ─── Helpers ────────────────────────────────────────────────
  function money(cents) {
    return "€" + (cents / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function escHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    return d.toLocaleDateString("en-GB");
  }

  function statusBadge(status) {
    const map = {
      active: "success", inactive: "secondary",
      DRAFT: "secondary", SENT: "info", RECEIVED: "success", CANCELLED: "danger",
      QUOTE: "secondary", CONFIRMED: "info", SHIPPED: "warning", INVOICED: "success",
      PAID: "success", OVERDUE: "danger",
    };
    return `<span class="badge badge-${map[status] || "secondary"}">${escHtml(status)}</span>`;
  }

  // ─── Toast ──────────────────────────────────────────────────
  function toast(msg, type = "success") {
    const el = document.createElement("div");
    el.className = "toast toast-" + type;
    el.textContent = msg;
    toastContainer.appendChild(el);
    setTimeout(() => {
      el.style.animation = "toastOut .3s ease forwards";
      setTimeout(() => el.remove(), 300);
    }, 3000);
  }

  // ─── Modal ──────────────────────────────────────────────────
  function openModal(title, body, footer, large) {
    modalTitle.textContent = title;
    modalBody.innerHTML = body;
    modalFooter.innerHTML = footer || "";
    const m = $("#modal");
    m.classList.toggle("modal-lg", !!large);
    modalOverlay.classList.remove("hidden");
  }

  function closeModal() {
    modalOverlay.classList.add("hidden");
  }

  modalClose.addEventListener("click", closeModal);
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) closeModal();
  });

  // ─── Confirm Dialog ─────────────────────────────────────────
  function confirmDialog(title, message) {
    confirmTitle.textContent = title;
    confirmBody.innerHTML = `<p>${escHtml(message)}</p>`;
    confirmOverlay.classList.remove("hidden");
    return new Promise((resolve) => {
      function yes() { cleanup(); resolve(true); }
      function no() { cleanup(); resolve(false); }
      function cleanup() {
        confirmOverlay.classList.add("hidden");
        confirmOk.removeEventListener("click", yes);
        confirmCancel.removeEventListener("click", no);
      }
      confirmOk.addEventListener("click", yes);
      confirmCancel.addEventListener("click", no);
    });
  }

  // ─── Pagination ─────────────────────────────────────────────
  function renderPagination(container, page, pages, cb) {
    if (pages <= 1) { container.innerHTML = ""; return; }
    let h = `<button ${page <= 1 ? "disabled" : ""} data-p="${page - 1}">‹ Prev</button>`;
    const start = Math.max(1, page - 2);
    const end = Math.min(pages, page + 2);
    for (let i = start; i <= end; i++) {
      h += `<button class="${i === page ? "active" : ""}" data-p="${i}">${i}</button>`;
    }
    h += `<span class="page-info">${page}/${pages}</span>`;
    h += `<button ${page >= pages ? "disabled" : ""} data-p="${page + 1}">Next ›</button>`;
    container.innerHTML = h;
    container.querySelectorAll("button[data-p]").forEach((b) =>
      b.addEventListener("click", () => cb(+b.dataset.p))
    );
  }

  // ─── Skeleton Loader ───────────────────────────────────────
  function skeleton(n = 5) {
    return Array(n).fill('<div class="skeleton skeleton-row"></div>').join("");
  }

  // ─── Auth ───────────────────────────────────────────────────
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.classList.add("hidden");
    const username = $("#username").value.trim();
    const password = $("#password").value;
    try {
      const data = await api("POST", "/auth/login", { username, password });
      token = data.access_token;
      localStorage.setItem("erp_token", token);
      await initApp();
    } catch (err) {
      loginError.textContent = err.message;
      loginError.classList.remove("hidden");
    }
  });

  function logout() {
    token = "";
    currentUser = null;
    localStorage.removeItem("erp_token");
    appShell.classList.add("hidden");
    loginScreen.classList.remove("hidden");
    mainContent.innerHTML = "";
    $("#username").value = "";
    $("#password").value = "";
  }

  logoutBtn.addEventListener("click", (e) => {
    e.preventDefault();
    logout();
  });

  async function initApp() {
    try {
      currentUser = await api("GET", "/auth/me");
    } catch {
      logout();
      return;
    }
    loginScreen.classList.add("hidden");
    appShell.classList.remove("hidden");
    userInfo.textContent = currentUser.full_name || currentUser.username;
    if (currentUser.is_default_password) {
      defaultPwWarn.classList.remove("hidden");
    } else {
      defaultPwWarn.classList.add("hidden");
    }
    route();
  }

  // ─── Sidebar Toggle ────────────────────────────────────────
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
    sidebar.classList.toggle("open");
  });

  // ─── Router ─────────────────────────────────────────────────
  const modules = {
    dashboard: renderDashboard,
    inventory: renderInventory,
    customers: renderCustomers,
    suppliers: renderSuppliers,
    "purchase-orders": renderPurchaseOrders,
    "sales-orders": renderSalesOrders,
    invoices: renderInvoices,
    accounting: renderAccounting,
    hr: renderHR,
    settings: renderSettings,
  };

  function route() {
    const hash = location.hash.replace("#", "") || "dashboard";
    const fn = modules[hash] || renderDashboard;
    $$(".nav-list a").forEach((a) => a.classList.toggle("active", a.dataset.module === hash));
    // close mobile sidebar
    sidebar.classList.remove("open");
    fn();
  }

  window.addEventListener("hashchange", () => {
    if (token) route();
  });

  $$(".nav-list a").forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      location.hash = a.dataset.module;
    })
  );

  // ═══════════════════════════════════════════════════════════════
  // DASHBOARD
  // ═══════════════════════════════════════════════════════════════
  async function renderDashboard() {
    mainContent.innerHTML = `
      <div class="page-header"><h2>📊 Dashboard</h2></div>
      <div class="kpi-grid">${skeleton(4)}</div>
      <div class="chart-container"><canvas id="revenue-chart"></canvas></div>
      <div class="card"><h3>Recent Activity</h3><div id="activity-feed">${skeleton(5)}</div></div>
    `;
    try {
      const d = await api("GET", "/api/dashboard");
      $(".kpi-grid").innerHTML = `
        <div class="kpi-card success">
          <div class="kpi-label">Total Revenue</div>
          <div class="kpi-value">${money(d.total_revenue)}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Open Orders</div>
          <div class="kpi-value">${d.open_orders}</div>
        </div>
        <div class="kpi-card warning">
          <div class="kpi-label">Low Stock Alerts</div>
          <div class="kpi-value">${d.low_stock_alerts}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Active Customers</div>
          <div class="kpi-value">${d.active_customers}</div>
        </div>
      `;

      // Revenue chart
      if (revenueChart) revenueChart.destroy();
      const ctx = document.getElementById("revenue-chart");
      if (ctx) {
        revenueChart = new Chart(ctx, {
          type: "bar",
          data: {
            labels: d.revenue_by_month.map((m) => m.month),
            datasets: [{
              label: "Revenue",
              data: d.revenue_by_month.map((m) => m.revenue / 100),
              backgroundColor: "rgba(0,180,216,0.6)",
              borderColor: "#00B4D8",
              borderWidth: 1,
            }],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#CCD6E8" } } },
            scales: {
              x: { ticks: { color: "#8892a4" }, grid: { color: "rgba(38,53,80,0.5)" } },
              y: { ticks: { color: "#8892a4", callback: (v) => "€" + v.toLocaleString() }, grid: { color: "rgba(38,53,80,0.5)" } },
            },
          },
        });
      }

      // Activity feed
      const feed = $("#activity-feed");
      if (d.recent_activity.length === 0) {
        feed.innerHTML = "<p>No recent activity.</p>";
      } else {
        feed.innerHTML = `<ul class="activity-feed">${d.recent_activity.map((a) =>
          `<li><span>${escHtml(a.description)}</span><span class="activity-time">${fmtDate(a.date)} · ${money(a.amount)}</span></li>`
        ).join("")}</ul>`;
      }
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // INVENTORY
  // ═══════════════════════════════════════════════════════════════
  let invPage = 1;
  let invSearch = "";

  async function renderInventory() {
    invPage = 1;
    invSearch = "";
    mainContent.innerHTML = `
      <div class="page-header">
        <h2>📦 Inventory</h2>
        <button class="btn btn-primary" id="add-product-btn">+ New Product</button>
      </div>
      <div class="table-controls">
        <input class="search-input" id="inv-search" placeholder="Search products…">
      </div>
      <div class="table-wrapper"><div id="inv-table">${skeleton(8)}</div></div>
      <div class="pagination" id="inv-pagination"></div>
    `;
    $("#add-product-btn").addEventListener("click", () => openProductForm());
    $("#inv-search").addEventListener("input", debounce((e) => {
      invSearch = e.target.value;
      invPage = 1;
      loadProducts();
    }, 300));
    loadProducts();
  }

  async function loadProducts() {
    try {
      const d = await api("GET", `/api/products?page=${invPage}&limit=20&search=${encodeURIComponent(invSearch)}`);
      const t = $("#inv-table");
      if (d.items.length === 0) {
        t.innerHTML = "<p>No products found.</p>";
      } else {
        t.innerHTML = `<table>
          <thead><tr><th>SKU</th><th>Name</th><th>Category</th><th>Qty</th><th>Price</th><th>Reorder</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((p) => `<tr>
            <td>${escHtml(p.sku)}</td>
            <td><a href="#" class="prod-detail" data-id="${p.id}">${escHtml(p.name)}</a></td>
            <td>${escHtml(p.category)}</td>
            <td class="${p.quantity <= p.reorder_level ? "negative" : ""}">${p.quantity}</td>
            <td>${money(p.unit_price)}</td>
            <td>${p.reorder_level}</td>
            <td class="btn-group">
              <button class="btn btn-sm btn-primary edit-prod" data-id="${p.id}">Edit</button>
              <button class="btn btn-sm btn-danger del-prod" data-id="${p.id}">Del</button>
            </td>
          </tr>`).join("")}</tbody></table>`;
        t.querySelectorAll(".prod-detail").forEach((a) =>
          a.addEventListener("click", (e) => { e.preventDefault(); showProductDetail(+a.dataset.id); })
        );
        t.querySelectorAll(".edit-prod").forEach((b) =>
          b.addEventListener("click", () => openProductForm(+b.dataset.id))
        );
        t.querySelectorAll(".del-prod").forEach((b) =>
          b.addEventListener("click", () => deleteProduct(+b.dataset.id))
        );
      }
      renderPagination($("#inv-pagination"), d.page, d.pages, (p) => { invPage = p; loadProducts(); });
    } catch (err) {
      toast(err.message, "error");
    }
  }

  function openProductForm(id) {
    const isEdit = !!id;
    const title = isEdit ? "Edit Product" : "New Product";
    const body = `
      <form id="product-form">
        <div class="form-row">
          <div class="form-group"><label>SKU</label><input id="pf-sku" required></div>
          <div class="form-group"><label>Name</label><input id="pf-name" required></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Category</label><input id="pf-category"></div>
          <div class="form-group"><label>Quantity</label><input type="number" id="pf-qty" min="0" value="0"></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Unit Price (cents)</label><input type="number" id="pf-price" min="0" value="0"></div>
          <div class="form-group"><label>Reorder Level</label><input type="number" id="pf-reorder" min="0" value="10"></div>
        </div>
      </form>
    `;
    const footer = `<button class="btn btn-secondary" id="pf-cancel">Cancel</button><button class="btn btn-primary" id="pf-save">Save</button>`;
    openModal(title, body, footer);

    if (isEdit) {
      api("GET", `/api/products/${id}`).then((p) => {
        $("#pf-sku").value = p.sku;
        $("#pf-name").value = p.name;
        $("#pf-category").value = p.category;
        $("#pf-qty").value = p.quantity;
        $("#pf-price").value = p.unit_price;
        $("#pf-reorder").value = p.reorder_level;
      }).catch((e) => toast(e.message, "error"));
    }

    $("#pf-cancel").addEventListener("click", closeModal);
    $("#pf-save").addEventListener("click", async () => {
      const data = {
        sku: $("#pf-sku").value.trim(),
        name: $("#pf-name").value.trim(),
        category: $("#pf-category").value.trim(),
        quantity: +$("#pf-qty").value,
        unit_price: +$("#pf-price").value,
        reorder_level: +$("#pf-reorder").value,
      };
      if (!data.sku || !data.name) { toast("SKU and Name are required", "error"); return; }
      try {
        if (isEdit) {
          await api("PUT", `/api/products/${id}`, data);
          toast("Product updated");
        } else {
          await api("POST", "/api/products", data);
          toast("Product created");
        }
        closeModal();
        loadProducts();
      } catch (err) {
        toast(err.message, "error");
      }
    });
  }

  async function deleteProduct(id) {
    if (await confirmDialog("Delete Product", "Are you sure you want to delete this product?")) {
      try {
        await api("DELETE", `/api/products/${id}`);
        toast("Product deleted");
        loadProducts();
      } catch (err) {
        toast(err.message, "error");
      }
    }
  }

  async function showProductDetail(id) {
    try {
      const p = await api("GET", `/api/products/${id}`);
      const body = `
        <div class="detail-grid">
          <div class="detail-item"><div class="detail-label">SKU</div><div class="detail-value">${escHtml(p.sku)}</div></div>
          <div class="detail-item"><div class="detail-label">Name</div><div class="detail-value">${escHtml(p.name)}</div></div>
          <div class="detail-item"><div class="detail-label">Category</div><div class="detail-value">${escHtml(p.category)}</div></div>
          <div class="detail-item"><div class="detail-label">Quantity</div><div class="detail-value">${p.quantity}</div></div>
          <div class="detail-item"><div class="detail-label">Unit Price</div><div class="detail-value">${money(p.unit_price)}</div></div>
          <div class="detail-item"><div class="detail-label">Reorder Level</div><div class="detail-value">${p.reorder_level}</div></div>
        </div>
        <h3>Stock Movements</h3>
        ${p.movements.length === 0 ? "<p>No movements yet.</p>" : `
        <div class="table-wrapper"><table>
          <thead><tr><th>Type</th><th>Qty</th><th>Reference</th><th>Notes</th><th>Date</th></tr></thead>
          <tbody>${p.movements.map((m) => `<tr>
            <td>${statusBadge(m.movement_type === "IN" ? "RECEIVED" : "SHIPPED")}</td>
            <td>${m.quantity}</td>
            <td>${escHtml(m.reference)}</td>
            <td>${escHtml(m.notes)}</td>
            <td>${fmtDate(m.created_at)}</td>
          </tr>`).join("")}</tbody>
        </table></div>`}
      `;
      openModal("Product Detail", body, "", true);
    } catch (err) {
      toast(err.message, "error");
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // CUSTOMERS
  // ═══════════════════════════════════════════════════════════════
  let custPage = 1, custSearch = "", custStatus = "";

  async function renderCustomers() {
    custPage = 1; custSearch = ""; custStatus = "";
    mainContent.innerHTML = `
      <div class="page-header">
        <h2>👥 Customers</h2>
        <button class="btn btn-primary" id="add-cust-btn">+ New Customer</button>
      </div>
      <div class="table-controls">
        <input class="search-input" id="cust-search" placeholder="Search customers…">
        <select id="cust-status-filter" class="search-input" style="min-width:150px">
          <option value="">All Status</option><option value="active">Active</option><option value="inactive">Inactive</option>
        </select>
      </div>
      <div class="table-wrapper"><div id="cust-table">${skeleton(8)}</div></div>
      <div class="pagination" id="cust-pagination"></div>
    `;
    $("#add-cust-btn").addEventListener("click", () => openCustomerForm());
    $("#cust-search").addEventListener("input", debounce((e) => { custSearch = e.target.value; custPage = 1; loadCustomers(); }, 300));
    $("#cust-status-filter").addEventListener("change", (e) => { custStatus = e.target.value; custPage = 1; loadCustomers(); });
    loadCustomers();
  }

  async function loadCustomers() {
    try {
      const d = await api("GET", `/api/customers?page=${custPage}&limit=20&search=${encodeURIComponent(custSearch)}&status=${custStatus}`);
      const t = $("#cust-table");
      if (d.items.length === 0) {
        t.innerHTML = "<p>No customers found.</p>";
      } else {
        t.innerHTML = `<table>
          <thead><tr><th>Name</th><th>Email</th><th>Phone</th><th>Company</th><th>Status</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((c) => `<tr>
            <td><a href="#" class="cust-detail" data-id="${c.id}">${escHtml(c.name)}</a></td>
            <td>${escHtml(c.email)}</td><td>${escHtml(c.phone)}</td><td>${escHtml(c.company)}</td>
            <td>${statusBadge(c.status)}</td>
            <td class="btn-group">
              <button class="btn btn-sm btn-primary edit-cust" data-id="${c.id}">Edit</button>
              <button class="btn btn-sm btn-danger del-cust" data-id="${c.id}">Del</button>
            </td>
          </tr>`).join("")}</tbody></table>`;
        t.querySelectorAll(".cust-detail").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); showCustomerDetail(+a.dataset.id); }));
        t.querySelectorAll(".edit-cust").forEach((b) => b.addEventListener("click", () => openCustomerForm(+b.dataset.id)));
        t.querySelectorAll(".del-cust").forEach((b) => b.addEventListener("click", () => deleteCustomer(+b.dataset.id)));
      }
      renderPagination($("#cust-pagination"), d.page, d.pages, (p) => { custPage = p; loadCustomers(); });
    } catch (err) { toast(err.message, "error"); }
  }

  function openCustomerForm(id) {
    const isEdit = !!id;
    const body = `<form id="cust-form">
      <div class="form-group"><label>Name</label><input id="cf-name" required></div>
      <div class="form-row">
        <div class="form-group"><label>Email</label><input type="email" id="cf-email"></div>
        <div class="form-group"><label>Phone</label><input id="cf-phone"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Company</label><input id="cf-company"></div>
        <div class="form-group"><label>Status</label><select id="cf-status"><option value="active">Active</option><option value="inactive">Inactive</option></select></div>
      </div>
      <div class="form-group"><label>Address</label><textarea id="cf-address"></textarea></div>
    </form>`;
    const footer = `<button class="btn btn-secondary" id="cf-cancel">Cancel</button><button class="btn btn-primary" id="cf-save">Save</button>`;
    openModal(isEdit ? "Edit Customer" : "New Customer", body, footer);

    if (isEdit) {
      api("GET", `/api/customers/${id}`).then((c) => {
        $("#cf-name").value = c.name; $("#cf-email").value = c.email; $("#cf-phone").value = c.phone;
        $("#cf-company").value = c.company; $("#cf-status").value = c.status; $("#cf-address").value = c.address;
      }).catch((e) => toast(e.message, "error"));
    }

    $("#cf-cancel").addEventListener("click", closeModal);
    $("#cf-save").addEventListener("click", async () => {
      const data = {
        name: $("#cf-name").value.trim(), email: $("#cf-email").value.trim(),
        phone: $("#cf-phone").value.trim(), company: $("#cf-company").value.trim(),
        status: $("#cf-status").value, address: $("#cf-address").value.trim(),
      };
      if (!data.name) { toast("Name is required", "error"); return; }
      try {
        if (isEdit) { await api("PUT", `/api/customers/${id}`, data); toast("Customer updated"); }
        else { await api("POST", "/api/customers", data); toast("Customer created"); }
        closeModal(); loadCustomers();
      } catch (err) { toast(err.message, "error"); }
    });
  }

  async function deleteCustomer(id) {
    if (await confirmDialog("Delete Customer", "Are you sure?")) {
      try { await api("DELETE", `/api/customers/${id}`); toast("Customer deleted"); loadCustomers(); }
      catch (err) { toast(err.message, "error"); }
    }
  }

  async function showCustomerDetail(id) {
    try {
      const c = await api("GET", `/api/customers/${id}`);
      const body = `
        <div class="detail-grid">
          <div class="detail-item"><div class="detail-label">Name</div><div class="detail-value">${escHtml(c.name)}</div></div>
          <div class="detail-item"><div class="detail-label">Email</div><div class="detail-value">${escHtml(c.email)}</div></div>
          <div class="detail-item"><div class="detail-label">Phone</div><div class="detail-value">${escHtml(c.phone)}</div></div>
          <div class="detail-item"><div class="detail-label">Company</div><div class="detail-value">${escHtml(c.company)}</div></div>
          <div class="detail-item"><div class="detail-label">Status</div><div class="detail-value">${statusBadge(c.status)}</div></div>
          <div class="detail-item"><div class="detail-label">Total Spent</div><div class="detail-value">${money(c.total_spent)}</div></div>
        </div>
        <h3>Order History</h3>
        ${c.orders.length === 0 ? "<p>No orders yet.</p>" : `
        <div class="table-wrapper"><table>
          <thead><tr><th>Order #</th><th>Status</th><th>Total</th><th>Date</th></tr></thead>
          <tbody>${c.orders.map((o) => `<tr>
            <td>${escHtml(o.order_number)}</td><td>${statusBadge(o.status)}</td>
            <td>${money(o.total_amount)}</td><td>${fmtDate(o.created_at)}</td>
          </tr>`).join("")}</tbody></table></div>`}
      `;
      openModal("Customer Detail", body, "", true);
    } catch (err) { toast(err.message, "error"); }
  }

  // ═══════════════════════════════════════════════════════════════
  // SUPPLIERS
  // ═══════════════════════════════════════════════════════════════
  let suppPage = 1, suppSearch = "";

  async function renderSuppliers() {
    suppPage = 1; suppSearch = "";
    mainContent.innerHTML = `
      <div class="page-header"><h2>🏭 Suppliers</h2>
        <button class="btn btn-primary" id="add-supp-btn">+ New Supplier</button></div>
      <div class="table-controls"><input class="search-input" id="supp-search" placeholder="Search suppliers…"></div>
      <div class="table-wrapper"><div id="supp-table">${skeleton(8)}</div></div>
      <div class="pagination" id="supp-pagination"></div>
    `;
    $("#add-supp-btn").addEventListener("click", () => openSupplierForm());
    $("#supp-search").addEventListener("input", debounce((e) => { suppSearch = e.target.value; suppPage = 1; loadSuppliers(); }, 300));
    loadSuppliers();
  }

  async function loadSuppliers() {
    try {
      const d = await api("GET", `/api/suppliers?page=${suppPage}&limit=20&search=${encodeURIComponent(suppSearch)}`);
      const t = $("#supp-table");
      if (d.items.length === 0) { t.innerHTML = "<p>No suppliers found.</p>"; }
      else {
        t.innerHTML = `<table>
          <thead><tr><th>Name</th><th>Contact</th><th>Email</th><th>Phone</th><th>Payment Terms</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((s) => `<tr>
            <td>${escHtml(s.name)}</td><td>${escHtml(s.contact_person)}</td><td>${escHtml(s.email)}</td>
            <td>${escHtml(s.phone)}</td><td>${escHtml(s.payment_terms)}</td>
            <td class="btn-group">
              <button class="btn btn-sm btn-primary edit-supp" data-id="${s.id}">Edit</button>
              <button class="btn btn-sm btn-danger del-supp" data-id="${s.id}">Del</button>
            </td>
          </tr>`).join("")}</tbody></table>`;
        t.querySelectorAll(".edit-supp").forEach((b) => b.addEventListener("click", () => openSupplierForm(+b.dataset.id)));
        t.querySelectorAll(".del-supp").forEach((b) => b.addEventListener("click", () => deleteSupplier(+b.dataset.id)));
      }
      renderPagination($("#supp-pagination"), d.page, d.pages, (p) => { suppPage = p; loadSuppliers(); });
    } catch (err) { toast(err.message, "error"); }
  }

  function openSupplierForm(id) {
    const isEdit = !!id;
    const body = `<form id="supp-form">
      <div class="form-group"><label>Name</label><input id="sf-name" required></div>
      <div class="form-row">
        <div class="form-group"><label>Contact Person</label><input id="sf-contact"></div>
        <div class="form-group"><label>Email</label><input type="email" id="sf-email"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Phone</label><input id="sf-phone"></div>
        <div class="form-group"><label>Payment Terms</label><input id="sf-terms" value="Net 30"></div>
      </div>
      <div class="form-group"><label>Address</label><textarea id="sf-address"></textarea></div>
    </form>`;
    const footer = `<button class="btn btn-secondary" id="sf-cancel">Cancel</button><button class="btn btn-primary" id="sf-save">Save</button>`;
    openModal(isEdit ? "Edit Supplier" : "New Supplier", body, footer);

    if (isEdit) {
      api("GET", `/api/suppliers/${id}`).then((s) => {
        $("#sf-name").value = s.name; $("#sf-contact").value = s.contact_person;
        $("#sf-email").value = s.email; $("#sf-phone").value = s.phone;
        $("#sf-terms").value = s.payment_terms; $("#sf-address").value = s.address;
      }).catch((e) => toast(e.message, "error"));
    }

    $("#sf-cancel").addEventListener("click", closeModal);
    $("#sf-save").addEventListener("click", async () => {
      const data = {
        name: $("#sf-name").value.trim(), contact_person: $("#sf-contact").value.trim(),
        email: $("#sf-email").value.trim(), phone: $("#sf-phone").value.trim(),
        payment_terms: $("#sf-terms").value.trim(), address: $("#sf-address").value.trim(),
      };
      if (!data.name) { toast("Name is required", "error"); return; }
      try {
        if (isEdit) { await api("PUT", `/api/suppliers/${id}`, data); toast("Supplier updated"); }
        else { await api("POST", "/api/suppliers", data); toast("Supplier created"); }
        closeModal(); loadSuppliers();
      } catch (err) { toast(err.message, "error"); }
    });
  }

  async function deleteSupplier(id) {
    if (await confirmDialog("Delete Supplier", "Are you sure?")) {
      try { await api("DELETE", `/api/suppliers/${id}`); toast("Supplier deleted"); loadSuppliers(); }
      catch (err) { toast(err.message, "error"); }
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // PURCHASE ORDERS
  // ═══════════════════════════════════════════════════════════════
  let poPage = 1, poSearch = "", poStatus = "";

  async function renderPurchaseOrders() {
    poPage = 1; poSearch = ""; poStatus = "";
    mainContent.innerHTML = `
      <div class="page-header"><h2>🛒 Purchase Orders</h2>
        <button class="btn btn-primary" id="add-po-btn">+ New PO</button></div>
      <div class="table-controls">
        <input class="search-input" id="po-search" placeholder="Search PO #…">
        <select id="po-status-filter" class="search-input" style="min-width:150px">
          <option value="">All Status</option><option value="DRAFT">Draft</option><option value="SENT">Sent</option>
          <option value="RECEIVED">Received</option><option value="CANCELLED">Cancelled</option>
        </select>
      </div>
      <div class="table-wrapper"><div id="po-table">${skeleton(8)}</div></div>
      <div class="pagination" id="po-pagination"></div>
    `;
    $("#add-po-btn").addEventListener("click", () => openPOForm());
    $("#po-search").addEventListener("input", debounce((e) => { poSearch = e.target.value; poPage = 1; loadPOs(); }, 300));
    $("#po-status-filter").addEventListener("change", (e) => { poStatus = e.target.value; poPage = 1; loadPOs(); });
    loadPOs();
  }

  async function loadPOs() {
    try {
      const d = await api("GET", `/api/purchase-orders?page=${poPage}&limit=20&search=${encodeURIComponent(poSearch)}&status=${poStatus}`);
      const t = $("#po-table");
      if (d.items.length === 0) { t.innerHTML = "<p>No purchase orders found.</p>"; }
      else {
        t.innerHTML = `<table>
          <thead><tr><th>PO #</th><th>Supplier</th><th>Items</th><th>Total</th><th>Status</th><th>Date</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((po) => `<tr>
            <td><a href="#" class="po-detail" data-id="${po.id}">${escHtml(po.order_number)}</a></td>
            <td>${escHtml(po.supplier_name)}</td><td>${po.item_count}</td><td>${money(po.total_amount)}</td>
            <td>${statusBadge(po.status)}</td><td>${fmtDate(po.created_at)}</td>
            <td class="btn-group">
              ${po.status === "DRAFT" ? `<button class="btn btn-sm btn-primary edit-po" data-id="${po.id}">Edit</button>` : ""}
              ${po.status === "DRAFT" ? `<button class="btn btn-sm btn-info po-send" data-id="${po.id}">Send</button>` : ""}
              ${po.status === "SENT" ? `<button class="btn btn-sm btn-success po-receive" data-id="${po.id}">Receive</button>` : ""}
              ${["DRAFT", "SENT"].includes(po.status) ? `<button class="btn btn-sm btn-danger po-cancel" data-id="${po.id}">Cancel</button>` : ""}
              <button class="btn btn-sm btn-danger del-po" data-id="${po.id}">Del</button>
            </td>
          </tr>`).join("")}</tbody></table>`;
        t.querySelectorAll(".po-detail").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); showPODetail(+a.dataset.id); }));
        t.querySelectorAll(".edit-po").forEach((b) => b.addEventListener("click", () => openPOForm(+b.dataset.id)));
        t.querySelectorAll(".po-send").forEach((b) => b.addEventListener("click", () => changePOStatus(+b.dataset.id, "SENT")));
        t.querySelectorAll(".po-receive").forEach((b) => b.addEventListener("click", () => changePOStatus(+b.dataset.id, "RECEIVED")));
        t.querySelectorAll(".po-cancel").forEach((b) => b.addEventListener("click", () => changePOStatus(+b.dataset.id, "CANCELLED")));
        t.querySelectorAll(".del-po").forEach((b) => b.addEventListener("click", () => deletePO(+b.dataset.id)));
      }
      renderPagination($("#po-pagination"), d.page, d.pages, (p) => { poPage = p; loadPOs(); });
    } catch (err) { toast(err.message, "error"); }
  }

  async function changePOStatus(id, status) {
    try { await api("PATCH", `/api/purchase-orders/${id}`, { status }); toast(`PO marked as ${status}`); loadPOs(); }
    catch (err) { toast(err.message, "error"); }
  }

  async function deletePO(id) {
    if (await confirmDialog("Delete PO", "Are you sure?")) {
      try { await api("DELETE", `/api/purchase-orders/${id}`); toast("PO deleted"); loadPOs(); }
      catch (err) { toast(err.message, "error"); }
    }
  }

  async function showPODetail(id) {
    try {
      const po = await api("GET", `/api/purchase-orders/${id}`);
      const body = `
        <div class="detail-grid">
          <div class="detail-item"><div class="detail-label">PO Number</div><div class="detail-value">${escHtml(po.order_number)}</div></div>
          <div class="detail-item"><div class="detail-label">Supplier</div><div class="detail-value">${escHtml(po.supplier_name)}</div></div>
          <div class="detail-item"><div class="detail-label">Status</div><div class="detail-value">${statusBadge(po.status)}</div></div>
          <div class="detail-item"><div class="detail-label">Total</div><div class="detail-value">${money(po.total_amount)}</div></div>
          <div class="detail-item"><div class="detail-label">Date</div><div class="detail-value">${fmtDate(po.created_at)}</div></div>
        </div>
        ${po.notes ? `<p><strong>Notes:</strong> ${escHtml(po.notes)}</p>` : ""}
        <h3>Line Items</h3>
        <div class="table-wrapper"><table>
          <thead><tr><th>Product</th><th>SKU</th><th>Qty</th><th>Unit Price</th><th>Total</th></tr></thead>
          <tbody>${(po.items || []).map((i) => `<tr>
            <td>${escHtml(i.product_name)}</td><td>${escHtml(i.product_sku)}</td>
            <td>${i.quantity}</td><td>${money(i.unit_price)}</td><td>${money(i.quantity * i.unit_price)}</td>
          </tr>`).join("")}</tbody>
        </table></div>
      `;
      openModal("Purchase Order Detail", body, "", true);
    } catch (err) { toast(err.message, "error"); }
  }

  async function openPOForm(id) {
    const isEdit = !!id;
    let suppliers = [], products = [];
    try {
      const [sData, pData] = await Promise.all([
        api("GET", "/api/suppliers?page=1&limit=200"),
        api("GET", "/api/products?page=1&limit=200"),
      ]);
      suppliers = sData.items;
      products = pData.items;
    } catch (err) { toast(err.message, "error"); return; }

    const suppOpts = suppliers.map((s) => `<option value="${s.id}">${escHtml(s.name)}</option>`).join("");
    const prodOpts = products.map((p) => `<option value="${p.id}" data-price="${p.unit_price}">${escHtml(p.name)} (${escHtml(p.sku)})</option>`).join("");
    const body = `
      <form id="po-form">
        <div class="form-group"><label>Supplier</label><select id="pof-supplier" required>${suppOpts}</select></div>
        <div class="form-group"><label>Notes</label><textarea id="pof-notes"></textarea></div>
        <h4>Line Items</h4>
        <div id="pof-items"></div>
        <button type="button" class="btn btn-sm btn-secondary" id="pof-add-item">+ Add Item</button>
      </form>`;
    const footer = `<button class="btn btn-secondary" id="pof-cancel">Cancel</button><button class="btn btn-primary" id="pof-save">Save</button>`;
    openModal(isEdit ? "Edit Purchase Order" : "New Purchase Order", body, footer, true);

    let items = [{ product_id: products[0]?.id || 0, quantity: 1, unit_price: Math.round((products[0]?.unit_price || 0) * 0.6) }];

    function renderItems() {
      const c = $("#pof-items");
      c.innerHTML = items.map((it, idx) => `
        <div class="line-item-row">
          <select class="pof-prod" data-idx="${idx}">${prodOpts}</select>
          <input type="number" class="pof-qty" data-idx="${idx}" min="1" value="${it.quantity}" placeholder="Qty">
          <input type="number" class="pof-price" data-idx="${idx}" min="0" value="${it.unit_price}" placeholder="Unit Price (cents)">
          <span>${money(it.quantity * it.unit_price)}</span>
          <button type="button" class="btn btn-sm btn-danger pof-remove" data-idx="${idx}">✕</button>
        </div>
      `).join("");
      c.querySelectorAll(".pof-prod").forEach((s) => { s.value = items[+s.dataset.idx].product_id; s.addEventListener("change", (e) => { items[+e.target.dataset.idx].product_id = +e.target.value; }); });
      c.querySelectorAll(".pof-qty").forEach((i) => i.addEventListener("input", (e) => { items[+e.target.dataset.idx].quantity = +e.target.value || 1; renderItems(); }));
      c.querySelectorAll(".pof-price").forEach((i) => i.addEventListener("input", (e) => { items[+e.target.dataset.idx].unit_price = +e.target.value || 0; renderItems(); }));
      c.querySelectorAll(".pof-remove").forEach((b) => b.addEventListener("click", () => { items.splice(+b.dataset.idx, 1); renderItems(); }));
    }

    if (isEdit) {
      api("GET", `/api/purchase-orders/${id}`).then((po) => {
        $("#pof-supplier").value = po.supplier_id;
        $("#pof-notes").value = po.notes;
        items = (po.items || []).map((i) => ({ product_id: i.product_id, quantity: i.quantity, unit_price: i.unit_price }));
        if (items.length === 0) items = [{ product_id: products[0]?.id || 0, quantity: 1, unit_price: 0 }];
        renderItems();
      }).catch((e) => toast(e.message, "error"));
    } else {
      renderItems();
    }

    $("#pof-add-item").addEventListener("click", () => { items.push({ product_id: products[0]?.id || 0, quantity: 1, unit_price: 0 }); renderItems(); });
    $("#pof-cancel").addEventListener("click", closeModal);
    $("#pof-save").addEventListener("click", async () => {
      const data = {
        supplier_id: +$("#pof-supplier").value,
        notes: $("#pof-notes").value.trim(),
        items: items.filter((i) => i.product_id && i.quantity > 0),
      };
      if (data.items.length === 0) { toast("Add at least one item", "error"); return; }
      try {
        if (isEdit) { await api("PUT", `/api/purchase-orders/${id}`, data); toast("PO updated"); }
        else { await api("POST", "/api/purchase-orders", data); toast("PO created"); }
        closeModal(); loadPOs();
      } catch (err) { toast(err.message, "error"); }
    });
  }

  // ═══════════════════════════════════════════════════════════════
  // SALES ORDERS
  // ═══════════════════════════════════════════════════════════════
  let soPage = 1, soSearch = "", soStatus = "";

  async function renderSalesOrders() {
    soPage = 1; soSearch = ""; soStatus = "";
    mainContent.innerHTML = `
      <div class="page-header"><h2>💼 Sales Orders</h2>
        <button class="btn btn-primary" id="add-so-btn">+ New Sales Order</button></div>
      <div class="table-controls">
        <input class="search-input" id="so-search" placeholder="Search SO #…">
        <select id="so-status-filter" class="search-input" style="min-width:150px">
          <option value="">All Status</option><option value="QUOTE">Quote</option><option value="CONFIRMED">Confirmed</option>
          <option value="SHIPPED">Shipped</option><option value="INVOICED">Invoiced</option><option value="CANCELLED">Cancelled</option>
        </select>
      </div>
      <div class="table-wrapper"><div id="so-table">${skeleton(8)}</div></div>
      <div class="pagination" id="so-pagination"></div>
    `;
    $("#add-so-btn").addEventListener("click", () => openSOForm());
    $("#so-search").addEventListener("input", debounce((e) => { soSearch = e.target.value; soPage = 1; loadSOs(); }, 300));
    $("#so-status-filter").addEventListener("change", (e) => { soStatus = e.target.value; soPage = 1; loadSOs(); });
    loadSOs();
  }

  async function loadSOs() {
    try {
      const d = await api("GET", `/api/sales-orders?page=${soPage}&limit=20&search=${encodeURIComponent(soSearch)}&status=${soStatus}`);
      const t = $("#so-table");
      if (d.items.length === 0) { t.innerHTML = "<p>No sales orders found.</p>"; }
      else {
        t.innerHTML = `<table>
          <thead><tr><th>SO #</th><th>Customer</th><th>Items</th><th>Subtotal</th><th>Tax</th><th>Total</th><th>Status</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((so) => `<tr>
            <td><a href="#" class="so-detail" data-id="${so.id}">${escHtml(so.order_number)}</a></td>
            <td>${escHtml(so.customer_name)}</td><td>${so.item_count}</td>
            <td>${money(so.subtotal)}</td><td>${money(so.tax_amount)}</td><td>${money(so.total_amount)}</td>
            <td>${statusBadge(so.status)}</td>
            <td class="btn-group">
              ${so.status === "QUOTE" ? `<button class="btn btn-sm btn-info so-confirm" data-id="${so.id}">Confirm</button>` : ""}
              ${so.status === "CONFIRMED" ? `<button class="btn btn-sm btn-warning so-ship" data-id="${so.id}">Ship</button>` : ""}
              ${["CONFIRMED", "SHIPPED"].includes(so.status) ? `<button class="btn btn-sm btn-success so-invoice" data-id="${so.id}">Invoice</button>` : ""}
              ${["QUOTE", "CONFIRMED"].includes(so.status) ? `<button class="btn btn-sm btn-danger so-cancel" data-id="${so.id}">Cancel</button>` : ""}
              <button class="btn btn-sm btn-danger del-so" data-id="${so.id}">Del</button>
            </td>
          </tr>`).join("")}</tbody></table>`;
        t.querySelectorAll(".so-detail").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); showSODetail(+a.dataset.id); }));
        t.querySelectorAll(".so-confirm").forEach((b) => b.addEventListener("click", () => changeSOStatus(+b.dataset.id, "CONFIRMED")));
        t.querySelectorAll(".so-ship").forEach((b) => b.addEventListener("click", () => changeSOStatus(+b.dataset.id, "SHIPPED")));
        t.querySelectorAll(".so-invoice").forEach((b) => b.addEventListener("click", () => createInvoiceFromSO(+b.dataset.id)));
        t.querySelectorAll(".so-cancel").forEach((b) => b.addEventListener("click", () => changeSOStatus(+b.dataset.id, "CANCELLED")));
        t.querySelectorAll(".del-so").forEach((b) => b.addEventListener("click", () => deleteSO(+b.dataset.id)));
      }
      renderPagination($("#so-pagination"), d.page, d.pages, (p) => { soPage = p; loadSOs(); });
    } catch (err) { toast(err.message, "error"); }
  }

  async function changeSOStatus(id, status) {
    try { await api("PATCH", `/api/sales-orders/${id}`, { status }); toast(`SO marked as ${status}`); loadSOs(); }
    catch (err) { toast(err.message, "error"); }
  }

  async function createInvoiceFromSO(soId) {
    try { await api("POST", "/api/invoices", { sales_order_id: soId }); toast("Invoice created"); loadSOs(); }
    catch (err) { toast(err.message, "error"); }
  }

  async function deleteSO(id) {
    if (await confirmDialog("Delete Sales Order", "Are you sure?")) {
      try { await api("DELETE", `/api/sales-orders/${id}`); toast("SO deleted"); loadSOs(); }
      catch (err) { toast(err.message, "error"); }
    }
  }

  async function showSODetail(id) {
    try {
      const so = await api("GET", `/api/sales-orders/${id}`);
      const body = `
        <div class="detail-grid">
          <div class="detail-item"><div class="detail-label">SO Number</div><div class="detail-value">${escHtml(so.order_number)}</div></div>
          <div class="detail-item"><div class="detail-label">Customer</div><div class="detail-value">${escHtml(so.customer_name)}</div></div>
          <div class="detail-item"><div class="detail-label">Status</div><div class="detail-value">${statusBadge(so.status)}</div></div>
          <div class="detail-item"><div class="detail-label">Date</div><div class="detail-value">${fmtDate(so.created_at)}</div></div>
        </div>
        ${so.notes ? `<p><strong>Notes:</strong> ${escHtml(so.notes)}</p>` : ""}
        <h3>Line Items</h3>
        <div class="table-wrapper"><table>
          <thead><tr><th>Product</th><th>SKU</th><th>Qty</th><th>Unit Price</th><th>Total</th></tr></thead>
          <tbody>${(so.items || []).map((i) => `<tr>
            <td>${escHtml(i.product_name)}</td><td>${escHtml(i.product_sku)}</td>
            <td>${i.quantity}</td><td>${money(i.unit_price)}</td><td>${money(i.quantity * i.unit_price)}</td>
          </tr>`).join("")}</tbody>
        </table></div>
        <div class="summary-box">
          <div class="summary-row"><span>Subtotal</span><span>${money(so.subtotal)}</span></div>
          <div class="summary-row"><span>Tax (${so.tax_rate}%)</span><span>${money(so.tax_amount)}</span></div>
          <div class="summary-row total"><span>Total</span><span>${money(so.total_amount)}</span></div>
        </div>
      `;
      openModal("Sales Order Detail", body, "", true);
    } catch (err) { toast(err.message, "error"); }
  }

  async function openSOForm(id) {
    const isEdit = !!id;
    let customers = [], products = [];
    try {
      const [cData, pData] = await Promise.all([
        api("GET", "/api/customers?page=1&limit=200&status=active"),
        api("GET", "/api/products?page=1&limit=200"),
      ]);
      customers = cData.items;
      products = pData.items;
    } catch (err) { toast(err.message, "error"); return; }

    const custOpts = customers.map((c) => `<option value="${c.id}">${escHtml(c.name)}</option>`).join("");
    const prodOpts = products.map((p) => `<option value="${p.id}" data-price="${p.unit_price}">${escHtml(p.name)} (${escHtml(p.sku)})</option>`).join("");
    const body = `
      <form id="so-form">
        <div class="form-row">
          <div class="form-group"><label>Customer</label><select id="sof-customer" required>${custOpts}</select></div>
          <div class="form-group"><label>Tax Rate (%)</label><input type="number" id="sof-tax" value="22" min="0" max="100"></div>
        </div>
        <div class="form-group"><label>Notes</label><textarea id="sof-notes"></textarea></div>
        <h4>Line Items</h4>
        <div id="sof-items"></div>
        <button type="button" class="btn btn-sm btn-secondary" id="sof-add-item">+ Add Item</button>
        <div class="summary-box mt-1" id="sof-summary"></div>
      </form>`;
    const footer = `<button class="btn btn-secondary" id="sof-cancel">Cancel</button><button class="btn btn-primary" id="sof-save">Save</button>`;
    openModal(isEdit ? "Edit Sales Order" : "New Sales Order", body, footer, true);

    let items = [{ product_id: products[0]?.id || 0, quantity: 1, unit_price: products[0]?.unit_price || 0 }];

    function renderItems() {
      const c = $("#sof-items");
      c.innerHTML = items.map((it, idx) => `
        <div class="line-item-row">
          <select class="sof-prod" data-idx="${idx}">${prodOpts}</select>
          <input type="number" class="sof-qty" data-idx="${idx}" min="1" value="${it.quantity}" placeholder="Qty">
          <input type="number" class="sof-price" data-idx="${idx}" min="0" value="${it.unit_price}" placeholder="Unit Price (cents)">
          <span>${money(it.quantity * it.unit_price)}</span>
          <button type="button" class="btn btn-sm btn-danger sof-remove" data-idx="${idx}">✕</button>
        </div>
      `).join("");
      c.querySelectorAll(".sof-prod").forEach((s) => {
        s.value = items[+s.dataset.idx].product_id;
        s.addEventListener("change", (e) => {
          const idx2 = +e.target.dataset.idx;
          items[idx2].product_id = +e.target.value;
          const opt = e.target.selectedOptions[0];
          if (opt) items[idx2].unit_price = +opt.dataset.price || 0;
          renderItems();
        });
      });
      c.querySelectorAll(".sof-qty").forEach((i) => i.addEventListener("input", (e) => { items[+e.target.dataset.idx].quantity = +e.target.value || 1; renderItems(); }));
      c.querySelectorAll(".sof-price").forEach((i) => i.addEventListener("input", (e) => { items[+e.target.dataset.idx].unit_price = +e.target.value || 0; renderItems(); }));
      c.querySelectorAll(".sof-remove").forEach((b) => b.addEventListener("click", () => { items.splice(+b.dataset.idx, 1); renderItems(); }));
      updateSOSummary();
    }

    function updateSOSummary() {
      const subtotal = items.reduce((s, i) => s + i.quantity * i.unit_price, 0);
      const taxRate = +( $("#sof-tax")?.value || 22);
      const tax = Math.round(subtotal * taxRate / 100);
      const total = subtotal + tax;
      const sum = $("#sof-summary");
      if (sum) {
        sum.innerHTML = `
          <div class="summary-row"><span>Subtotal</span><span>${money(subtotal)}</span></div>
          <div class="summary-row"><span>Tax (${taxRate}%)</span><span>${money(tax)}</span></div>
          <div class="summary-row total"><span>Total</span><span>${money(total)}</span></div>
        `;
      }
    }

    if (isEdit) {
      api("GET", `/api/sales-orders/${id}`).then((so) => {
        $("#sof-customer").value = so.customer_id;
        $("#sof-tax").value = so.tax_rate;
        $("#sof-notes").value = so.notes;
        items = (so.items || []).map((i) => ({ product_id: i.product_id, quantity: i.quantity, unit_price: i.unit_price }));
        if (items.length === 0) items = [{ product_id: products[0]?.id || 0, quantity: 1, unit_price: 0 }];
        renderItems();
      }).catch((e) => toast(e.message, "error"));
    } else {
      renderItems();
    }

    $("#sof-add-item").addEventListener("click", () => { items.push({ product_id: products[0]?.id || 0, quantity: 1, unit_price: products[0]?.unit_price || 0 }); renderItems(); });
    const taxInput = $("#sof-tax");
    if (taxInput) taxInput.addEventListener("input", () => updateSOSummary());
    $("#sof-cancel").addEventListener("click", closeModal);
    $("#sof-save").addEventListener("click", async () => {
      const data = {
        customer_id: +$("#sof-customer").value,
        tax_rate: +$("#sof-tax").value,
        notes: $("#sof-notes").value.trim(),
        items: items.filter((i) => i.product_id && i.quantity > 0),
      };
      if (data.items.length === 0) { toast("Add at least one item", "error"); return; }
      try {
        if (isEdit) { await api("PUT", `/api/sales-orders/${id}`, data); toast("SO updated"); }
        else { await api("POST", "/api/sales-orders", data); toast("SO created"); }
        closeModal(); loadSOs();
      } catch (err) { toast(err.message, "error"); }
    });
  }

  // ═══════════════════════════════════════════════════════════════
  // INVOICES
  // ═══════════════════════════════════════════════════════════════
  let invPageN = 1, invSearchN = "", invStatusN = "";

  async function renderInvoices() {
    invPageN = 1; invSearchN = ""; invStatusN = "";
    mainContent.innerHTML = `
      <div class="page-header"><h2>🧾 Invoices</h2></div>
      <div class="table-controls">
        <input class="search-input" id="inv-search-n" placeholder="Search invoice #…">
        <select id="inv-status-filter" class="search-input" style="min-width:150px">
          <option value="">All Status</option><option value="DRAFT">Draft</option><option value="SENT">Sent</option>
          <option value="PAID">Paid</option><option value="OVERDUE">Overdue</option>
        </select>
      </div>
      <div class="table-wrapper"><div id="inv-table-n">${skeleton(8)}</div></div>
      <div class="pagination" id="inv-pagination-n"></div>
    `;
    $("#inv-search-n").addEventListener("input", debounce((e) => { invSearchN = e.target.value; invPageN = 1; loadInvoices(); }, 300));
    $("#inv-status-filter").addEventListener("change", (e) => { invStatusN = e.target.value; invPageN = 1; loadInvoices(); });
    loadInvoices();
  }

  async function loadInvoices() {
    try {
      const d = await api("GET", `/api/invoices?page=${invPageN}&limit=20&search=${encodeURIComponent(invSearchN)}&status=${invStatusN}`);
      const t = $("#inv-table-n");
      if (d.items.length === 0) { t.innerHTML = "<p>No invoices found.</p>"; }
      else {
        t.innerHTML = `<table>
          <thead><tr><th>Invoice #</th><th>Customer</th><th>SO #</th><th>Subtotal</th><th>Tax</th><th>Total</th><th>Due</th><th>Status</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((inv) => `<tr>
            <td><a href="#" class="inv-detail" data-id="${inv.id}">${escHtml(inv.invoice_number)}</a></td>
            <td>${escHtml(inv.customer_name || "—")}</td><td>${escHtml(inv.order_number || "—")}</td>
            <td>${money(inv.subtotal)}</td><td>${money(inv.tax_amount)}</td><td>${money(inv.total_amount)}</td>
            <td>${fmtDate(inv.due_date)}</td><td>${statusBadge(inv.status)}</td>
            <td class="btn-group">
              ${inv.status === "DRAFT" ? `<button class="btn btn-sm btn-info inv-send" data-id="${inv.id}">Send</button>` : ""}
              ${inv.status === "SENT" || inv.status === "OVERDUE" ? `<button class="btn btn-sm btn-success inv-pay" data-id="${inv.id}">Mark Paid</button>` : ""}
              <button class="btn btn-sm btn-danger del-inv" data-id="${inv.id}">Del</button>
            </td>
          </tr>`).join("")}</tbody></table>`;
        t.querySelectorAll(".inv-detail").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); showInvoiceDetail(+a.dataset.id); }));
        t.querySelectorAll(".inv-send").forEach((b) => b.addEventListener("click", () => changeInvStatus(+b.dataset.id, "SENT")));
        t.querySelectorAll(".inv-pay").forEach((b) => b.addEventListener("click", () => changeInvStatus(+b.dataset.id, "PAID")));
        t.querySelectorAll(".del-inv").forEach((b) => b.addEventListener("click", () => deleteInvoice(+b.dataset.id)));
      }
      renderPagination($("#inv-pagination-n"), d.page, d.pages, (p) => { invPageN = p; loadInvoices(); });
    } catch (err) { toast(err.message, "error"); }
  }

  async function changeInvStatus(id, status) {
    try { await api("PATCH", `/api/invoices/${id}`, { status }); toast(`Invoice marked as ${status}`); loadInvoices(); }
    catch (err) { toast(err.message, "error"); }
  }

  async function deleteInvoice(id) {
    if (await confirmDialog("Delete Invoice", "Are you sure?")) {
      try { await api("DELETE", `/api/invoices/${id}`); toast("Invoice deleted"); loadInvoices(); }
      catch (err) { toast(err.message, "error"); }
    }
  }

  async function showInvoiceDetail(id) {
    try {
      const inv = await api("GET", `/api/invoices/${id}`);
      const body = `
        <div class="detail-grid">
          <div class="detail-item"><div class="detail-label">Invoice #</div><div class="detail-value">${escHtml(inv.invoice_number)}</div></div>
          <div class="detail-item"><div class="detail-label">Customer</div><div class="detail-value">${escHtml(inv.customer_name || "—")}</div></div>
          <div class="detail-item"><div class="detail-label">Sales Order</div><div class="detail-value">${escHtml(inv.order_number || "—")}</div></div>
          <div class="detail-item"><div class="detail-label">Status</div><div class="detail-value">${statusBadge(inv.status)}</div></div>
          <div class="detail-item"><div class="detail-label">Due Date</div><div class="detail-value">${fmtDate(inv.due_date)}</div></div>
          <div class="detail-item"><div class="detail-label">Created</div><div class="detail-value">${fmtDate(inv.created_at)}</div></div>
        </div>
        ${inv.notes ? `<p><strong>Notes:</strong> ${escHtml(inv.notes)}</p>` : ""}
        <h3>Line Items</h3>
        <div class="table-wrapper"><table>
          <thead><tr><th>Description</th><th>Qty</th><th>Unit Price</th><th>Amount</th></tr></thead>
          <tbody>${(inv.items || []).map((i) => `<tr>
            <td>${escHtml(i.description)}</td><td>${i.quantity}</td>
            <td>${money(i.unit_price)}</td><td>${money(i.amount)}</td>
          </tr>`).join("")}</tbody>
        </table></div>
        <div class="summary-box">
          <div class="summary-row"><span>Subtotal</span><span>${money(inv.subtotal)}</span></div>
          <div class="summary-row"><span>Tax</span><span>${money(inv.tax_amount)}</span></div>
          <div class="summary-row total"><span>Total</span><span>${money(inv.total_amount)}</span></div>
        </div>
      `;
      const footer2 = `<button class="btn btn-secondary no-print" onclick="window.print()">🖨 Print</button>`;
      openModal("Invoice Detail", body, footer2, true);
    } catch (err) { toast(err.message, "error"); }
  }

  // ═══════════════════════════════════════════════════════════════
  // ACCOUNTING
  // ═══════════════════════════════════════════════════════════════
  async function renderAccounting() {
    mainContent.innerHTML = `
      <div class="page-header"><h2>📒 Accounting</h2></div>
      <div class="tabs">
        <button class="tab-btn active" data-tab="accounts">Chart of Accounts</button>
        <button class="tab-btn" data-tab="journal">Journal Entries</button>
        <button class="tab-btn" data-tab="balance">Balance Sheet</button>
        <button class="tab-btn" data-tab="pl">Profit & Loss</button>
      </div>
      <div class="tab-content active" id="tab-accounts">${skeleton(5)}</div>
      <div class="tab-content" id="tab-journal">${skeleton(5)}</div>
      <div class="tab-content" id="tab-balance">${skeleton(5)}</div>
      <div class="tab-content" id="tab-pl">${skeleton(5)}</div>
    `;
    $$(".tab-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab-content").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        $(`#tab-${btn.dataset.tab}`).classList.add("active");
      })
    );
    loadAccounts();
    loadJournal();
    loadBalanceSheet();
    loadPL();
  }

  async function loadAccounts() {
    try {
      const d = await api("GET", "/api/accounts");
      const container = $("#tab-accounts");
      container.innerHTML = `
        <div class="flex-between mb-1">
          <h3>Accounts</h3>
          <button class="btn btn-primary btn-sm" id="add-account-btn">+ New Account</button>
        </div>
        <div class="table-wrapper"><table>
          <thead><tr><th>Code</th><th>Name</th><th>Type</th><th>Balance</th></tr></thead>
          <tbody>${d.items.map((a) => `<tr>
            <td>${escHtml(a.code)}</td><td>${escHtml(a.name)}</td>
            <td>${statusBadge(a.account_type)}</td>
            <td class="${a.balance >= 0 ? "positive" : "negative"}">${money(a.balance)}</td>
          </tr>`).join("")}</tbody>
        </table></div>
      `;
      $("#add-account-btn").addEventListener("click", openAccountForm);
    } catch (err) { toast(err.message, "error"); }
  }

  function openAccountForm() {
    const body = `<form id="acc-form">
      <div class="form-row">
        <div class="form-group"><label>Code</label><input id="acc-code" required placeholder="e.g. 1100"></div>
        <div class="form-group"><label>Name</label><input id="acc-name" required></div>
      </div>
      <div class="form-group"><label>Type</label><select id="acc-type">
        <option value="ASSET">Asset</option><option value="LIABILITY">Liability</option>
        <option value="EQUITY">Equity</option><option value="REVENUE">Revenue</option><option value="EXPENSE">Expense</option>
      </select></div>
    </form>`;
    const footer = `<button class="btn btn-secondary" id="acc-cancel">Cancel</button><button class="btn btn-primary" id="acc-save">Save</button>`;
    openModal("New Account", body, footer);
    $("#acc-cancel").addEventListener("click", closeModal);
    $("#acc-save").addEventListener("click", async () => {
      const data = { code: $("#acc-code").value.trim(), name: $("#acc-name").value.trim(), account_type: $("#acc-type").value };
      if (!data.code || !data.name) { toast("Code and Name required", "error"); return; }
      try { await api("POST", "/api/accounts", data); toast("Account created"); closeModal(); loadAccounts(); }
      catch (err) { toast(err.message, "error"); }
    });
  }

  let jePage = 1;
  async function loadJournal() {
    try {
      const d = await api("GET", `/api/journal-entries?page=${jePage}&limit=50`);
      const container = $("#tab-journal");
      container.innerHTML = `
        <div class="flex-between mb-1">
          <h3>Journal Entries</h3>
          <button class="btn btn-primary btn-sm" id="add-je-btn">+ New Entry</button>
        </div>
        <div class="table-wrapper"><table>
          <thead><tr><th>Date</th><th>Description</th><th>Debit Account</th><th>Credit Account</th><th>Amount</th></tr></thead>
          <tbody>${d.items.map((j) => `<tr>
            <td>${fmtDate(j.entry_date)}</td><td>${escHtml(j.description)}</td>
            <td>${escHtml(j.debit_account_name)}</td><td>${escHtml(j.credit_account_name)}</td>
            <td>${money(j.amount)}</td>
          </tr>`).join("")}</tbody>
        </table></div>
        <div class="pagination" id="je-pagination"></div>
      `;
      renderPagination($("#je-pagination"), d.page, d.pages, (p) => { jePage = p; loadJournal(); });
      $("#add-je-btn").addEventListener("click", openJEForm);
    } catch (err) { toast(err.message, "error"); }
  }

  async function openJEForm() {
    let accounts = [];
    try { accounts = (await api("GET", "/api/accounts")).items; } catch (e) { toast(e.message, "error"); return; }
    const accOpts = accounts.map((a) => `<option value="${a.id}">${escHtml(a.code)} — ${escHtml(a.name)}</option>`).join("");
    const today = new Date().toISOString().slice(0, 10);
    const body = `<form id="je-form">
      <div class="form-row">
        <div class="form-group"><label>Date</label><input type="date" id="je-date" value="${today}" required></div>
        <div class="form-group"><label>Amount (cents)</label><input type="number" id="je-amount" min="1" required></div>
      </div>
      <div class="form-group"><label>Description</label><input id="je-desc" required></div>
      <div class="form-row">
        <div class="form-group"><label>Debit Account</label><select id="je-debit">${accOpts}</select></div>
        <div class="form-group"><label>Credit Account</label><select id="je-credit">${accOpts}</select></div>
      </div>
    </form>`;
    const footer = `<button class="btn btn-secondary" id="je-cancel">Cancel</button><button class="btn btn-primary" id="je-save">Save</button>`;
    openModal("New Journal Entry", body, footer);
    $("#je-cancel").addEventListener("click", closeModal);
    $("#je-save").addEventListener("click", async () => {
      const data = {
        entry_date: $("#je-date").value, description: $("#je-desc").value.trim(),
        debit_account_id: +$("#je-debit").value, credit_account_id: +$("#je-credit").value,
        amount: +$("#je-amount").value,
      };
      if (!data.description || !data.amount) { toast("All fields required", "error"); return; }
      try { await api("POST", "/api/journal-entries", data); toast("Journal entry created"); closeModal(); loadJournal(); loadAccounts(); }
      catch (err) { toast(err.message, "error"); }
    });
  }

  async function loadBalanceSheet() {
    try {
      const d = await api("GET", "/api/balance-sheet");
      const container = $("#tab-balance");
      container.innerHTML = `
        <div class="ledger-grid">
          <div class="card">
            <h3>Assets</h3>
            ${d.assets.map((a) => `<div class="account-item"><span>${escHtml(a.code)} — ${escHtml(a.name)}</span><span class="positive">${money(a.balance)}</span></div>`).join("")}
            <div class="account-item" style="font-weight:700;border-top:2px solid var(--border);padding-top:.5rem;margin-top:.5rem"><span>Total Assets</span><span>${money(d.total_assets)}</span></div>
          </div>
          <div class="card">
            <h3>Liabilities</h3>
            ${d.liabilities.map((a) => `<div class="account-item"><span>${escHtml(a.code)} — ${escHtml(a.name)}</span><span>${money(a.balance)}</span></div>`).join("")}
            <div class="account-item" style="font-weight:700;border-top:2px solid var(--border);padding-top:.5rem;margin-top:.5rem"><span>Total Liabilities</span><span>${money(d.total_liabilities)}</span></div>
            <h3 class="mt-1">Equity</h3>
            ${d.equity.map((a) => `<div class="account-item"><span>${escHtml(a.code)} — ${escHtml(a.name)}</span><span>${money(a.balance)}</span></div>`).join("")}
            <div class="account-item" style="font-weight:700;border-top:2px solid var(--border);padding-top:.5rem;margin-top:.5rem"><span>Total Equity</span><span>${money(d.total_equity)}</span></div>
          </div>
        </div>
      `;
    } catch (err) { toast(err.message, "error"); }
  }

  async function loadPL() {
    try {
      const d = await api("GET", "/api/profit-loss");
      const container = $("#tab-pl");
      container.innerHTML = `
        <div class="kpi-grid mb-1">
          <div class="kpi-card success"><div class="kpi-label">Total Revenue</div><div class="kpi-value">${money(d.total_revenue)}</div></div>
          <div class="kpi-card danger"><div class="kpi-label">Total Expenses</div><div class="kpi-value">${money(d.total_expenses)}</div></div>
          <div class="kpi-card ${d.net_profit >= 0 ? "success" : "danger"}"><div class="kpi-label">Net Profit</div><div class="kpi-value">${money(d.net_profit)}</div></div>
        </div>
        <div class="chart-container"><canvas id="pl-chart"></canvas></div>
        <div class="ledger-grid">
          <div class="card">
            <h3>Revenue Accounts</h3>
            ${d.revenue_accounts.map((a) => `<div class="account-item"><span>${escHtml(a.name)}</span><span class="positive">${money(a.balance)}</span></div>`).join("")}
          </div>
          <div class="card">
            <h3>Expense Accounts</h3>
            ${d.expense_accounts.map((a) => `<div class="account-item"><span>${escHtml(a.name)}</span><span class="negative">${money(a.balance)}</span></div>`).join("")}
          </div>
        </div>
      `;
      // P&L chart
      if (plChart) plChart.destroy();
      const ctx = document.getElementById("pl-chart");
      if (ctx) {
        plChart = new Chart(ctx, {
          type: "bar",
          data: {
            labels: d.monthly.map((m) => m.month),
            datasets: [
              { label: "Revenue", data: d.monthly.map((m) => m.revenue / 100), backgroundColor: "rgba(46,204,113,0.6)", borderColor: "#2ecc71", borderWidth: 1 },
              { label: "Expenses", data: d.monthly.map((m) => m.expenses / 100), backgroundColor: "rgba(231,76,60,0.6)", borderColor: "#e74c3c", borderWidth: 1 },
            ],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#CCD6E8" } } },
            scales: {
              x: { ticks: { color: "#8892a4" }, grid: { color: "rgba(38,53,80,0.5)" } },
              y: { ticks: { color: "#8892a4", callback: (v) => "€" + v.toLocaleString() }, grid: { color: "rgba(38,53,80,0.5)" } },
            },
          },
        });
      }
    } catch (err) { toast(err.message, "error"); }
  }

  // ═══════════════════════════════════════════════════════════════
  // HR
  // ═══════════════════════════════════════════════════════════════
  let hrPage = 1, hrSearch = "", hrDept = "";

  async function renderHR() {
    hrPage = 1; hrSearch = ""; hrDept = "";
    mainContent.innerHTML = `
      <div class="page-header"><h2>👔 HR</h2>
        <button class="btn btn-primary" id="add-emp-btn">+ New Employee</button></div>
      <div class="tabs">
        <button class="tab-btn active" data-tab="emp-list">Employees</button>
        <button class="tab-btn" data-tab="payroll">Payroll Summary</button>
      </div>
      <div class="tab-content active" id="tab-emp-list">
        <div class="table-controls">
          <input class="search-input" id="hr-search" placeholder="Search employees…">
          <input class="search-input" id="hr-dept" placeholder="Filter department…" style="min-width:150px">
        </div>
        <div class="table-wrapper"><div id="hr-table">${skeleton(8)}</div></div>
        <div class="pagination" id="hr-pagination"></div>
      </div>
      <div class="tab-content" id="tab-payroll">${skeleton(5)}</div>
    `;
    $$(".tab-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab-content").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        $(`#tab-${btn.dataset.tab}`).classList.add("active");
      })
    );
    $("#add-emp-btn").addEventListener("click", () => openEmployeeForm());
    $("#hr-search").addEventListener("input", debounce((e) => { hrSearch = e.target.value; hrPage = 1; loadEmployees(); }, 300));
    $("#hr-dept").addEventListener("input", debounce((e) => { hrDept = e.target.value; hrPage = 1; loadEmployees(); }, 300));
    loadEmployees();
    loadPayroll();
  }

  async function loadEmployees() {
    try {
      const d = await api("GET", `/api/employees?page=${hrPage}&limit=20&search=${encodeURIComponent(hrSearch)}&department=${encodeURIComponent(hrDept)}`);
      const t = $("#hr-table");
      if (d.items.length === 0) { t.innerHTML = "<p>No employees found.</p>"; }
      else {
        t.innerHTML = `<table>
          <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Department</th><th>Salary/mo</th><th>Status</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((e) => `<tr>
            <td>${escHtml(e.name)}</td><td>${escHtml(e.email)}</td><td>${escHtml(e.role)}</td>
            <td>${escHtml(e.department)}</td><td>${money(e.salary)}</td><td>${statusBadge(e.status)}</td>
            <td class="btn-group">
              <button class="btn btn-sm btn-primary edit-emp" data-id="${e.id}">Edit</button>
              <button class="btn btn-sm btn-danger del-emp" data-id="${e.id}">Del</button>
            </td>
          </tr>`).join("")}</tbody></table>`;
        t.querySelectorAll(".edit-emp").forEach((b) => b.addEventListener("click", () => openEmployeeForm(+b.dataset.id)));
        t.querySelectorAll(".del-emp").forEach((b) => b.addEventListener("click", () => deleteEmployee(+b.dataset.id)));
      }
      renderPagination($("#hr-pagination"), d.page, d.pages, (p) => { hrPage = p; loadEmployees(); });
    } catch (err) { toast(err.message, "error"); }
  }

  function openEmployeeForm(id) {
    const isEdit = !!id;
    const today = new Date().toISOString().slice(0, 10);
    const body = `<form id="emp-form">
      <div class="form-group"><label>Name</label><input id="ef-name" required></div>
      <div class="form-row">
        <div class="form-group"><label>Email</label><input type="email" id="ef-email"></div>
        <div class="form-group"><label>Role</label><input id="ef-role"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Department</label><input id="ef-dept"></div>
        <div class="form-group"><label>Monthly Salary (cents)</label><input type="number" id="ef-salary" min="0" value="0"></div>
      </div>
      <div class="form-row">
        <div class="form-group"><label>Hire Date</label><input type="date" id="ef-hire" value="${today}"></div>
        <div class="form-group"><label>Status</label><select id="ef-status"><option value="active">Active</option><option value="inactive">Inactive</option></select></div>
      </div>
    </form>`;
    const footer = `<button class="btn btn-secondary" id="ef-cancel">Cancel</button><button class="btn btn-primary" id="ef-save">Save</button>`;
    openModal(isEdit ? "Edit Employee" : "New Employee", body, footer);

    if (isEdit) {
      api("GET", `/api/employees/${id}`).then((e) => {
        $("#ef-name").value = e.name; $("#ef-email").value = e.email; $("#ef-role").value = e.role;
        $("#ef-dept").value = e.department; $("#ef-salary").value = e.salary;
        $("#ef-hire").value = e.hire_date || today; $("#ef-status").value = e.status;
      }).catch((e) => toast(e.message, "error"));
    }

    $("#ef-cancel").addEventListener("click", closeModal);
    $("#ef-save").addEventListener("click", async () => {
      const data = {
        name: $("#ef-name").value.trim(), email: $("#ef-email").value.trim(),
        role: $("#ef-role").value.trim(), department: $("#ef-dept").value.trim(),
        salary: +$("#ef-salary").value, hire_date: $("#ef-hire").value, status: $("#ef-status").value,
      };
      if (!data.name) { toast("Name is required", "error"); return; }
      try {
        if (isEdit) { await api("PUT", `/api/employees/${id}`, data); toast("Employee updated"); }
        else { await api("POST", "/api/employees", data); toast("Employee created"); }
        closeModal(); loadEmployees(); loadPayroll();
      } catch (err) { toast(err.message, "error"); }
    });
  }

  async function deleteEmployee(id) {
    if (await confirmDialog("Delete Employee", "Are you sure?")) {
      try { await api("DELETE", `/api/employees/${id}`); toast("Employee removed"); loadEmployees(); loadPayroll(); }
      catch (err) { toast(err.message, "error"); }
    }
  }

  async function loadPayroll() {
    try {
      const d = await api("GET", "/api/payroll-summary");
      const container = $("#tab-payroll");
      container.innerHTML = `
        <div class="kpi-grid mb-1">
          <div class="kpi-card"><div class="kpi-label">Total Monthly Payroll</div><div class="kpi-value">${money(d.grand_total)}</div></div>
        </div>
        <div class="table-wrapper"><table>
          <thead><tr><th>Department</th><th>Employees</th><th>Total Salary/mo</th></tr></thead>
          <tbody>${d.departments.map((dp) => `<tr>
            <td>${escHtml(dp.department)}</td><td>${dp.employee_count}</td><td>${money(dp.total_salary)}</td>
          </tr>`).join("")}</tbody>
        </table></div>
      `;
    } catch (err) { toast(err.message, "error"); }
  }

  // ═══════════════════════════════════════════════════════════════
  // SETTINGS
  // ═══════════════════════════════════════════════════════════════
  async function renderSettings() {
    mainContent.innerHTML = `
      <div class="page-header"><h2>⚙️ Settings</h2></div>
      <div class="tabs">
        <button class="tab-btn active" data-tab="company">Company</button>
        <button class="tab-btn" data-tab="password">Change Password</button>
        ${currentUser?.role === "admin" ? '<button class="tab-btn" data-tab="users">Users</button>' : ""}
      </div>
      <div class="tab-content active" id="tab-company">${skeleton(5)}</div>
      <div class="tab-content" id="tab-password"></div>
      ${currentUser?.role === "admin" ? '<div class="tab-content" id="tab-users"></div>' : ""}
    `;
    $$(".tab-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab-content").forEach((c) => c.classList.remove("active"));
        btn.classList.add("active");
        $(`#tab-${btn.dataset.tab}`).classList.add("active");
      })
    );
    loadSettings();
    renderPasswordTab();
    if (currentUser?.role === "admin") loadUsers();
  }

  async function loadSettings() {
    try {
      const d = await api("GET", "/api/settings");
      const container = $("#tab-company");
      const isAdmin = currentUser?.role === "admin";
      container.innerHTML = `
        <div class="card">
          <h3>Company Information</h3>
          <form id="settings-form">
            <div class="form-group"><label>Company Name</label><input id="st-name" value="${escHtml(d.company_name || "")}" ${isAdmin ? "" : "disabled"}></div>
            <div class="form-group"><label>Address</label><textarea id="st-address" ${isAdmin ? "" : "disabled"}>${escHtml(d.company_address || "")}</textarea></div>
            <div class="form-row">
              <div class="form-group"><label>VAT Number</label><input id="st-vat" value="${escHtml(d.company_vat || "")}" ${isAdmin ? "" : "disabled"}></div>
              <div class="form-group"><label>Currency</label><input id="st-currency" value="${escHtml(d.currency || "EUR")}" ${isAdmin ? "" : "disabled"}></div>
            </div>
            <div class="form-row">
              <div class="form-group"><label>Currency Symbol</label><input id="st-symbol" value="${escHtml(d.currency_symbol || "€")}" ${isAdmin ? "" : "disabled"}></div>
              <div class="form-group"><label>Default Tax Rate (%)</label><input id="st-tax" value="${escHtml(d.default_tax_rate || "22")}" ${isAdmin ? "" : "disabled"}></div>
            </div>
            ${isAdmin ? '<button type="button" class="btn btn-primary" id="st-save">Save Settings</button>' : ""}
          </form>
        </div>
      `;
      if (isAdmin) {
        $("#st-save").addEventListener("click", async () => {
          const settings = [
            { key: "company_name", value: $("#st-name").value },
            { key: "company_address", value: $("#st-address").value },
            { key: "company_vat", value: $("#st-vat").value },
            { key: "currency", value: $("#st-currency").value },
            { key: "currency_symbol", value: $("#st-symbol").value },
            { key: "default_tax_rate", value: $("#st-tax").value },
          ];
          try { await api("PUT", "/api/settings", settings); toast("Settings saved"); }
          catch (err) { toast(err.message, "error"); }
        });
      }
    } catch (err) { toast(err.message, "error"); }
  }

  function renderPasswordTab() {
    const container = $("#tab-password");
    container.innerHTML = `
      <div class="card" style="max-width:400px">
        <h3>Change Password</h3>
        <form id="pw-form">
          <div class="form-group"><label>Current Password</label><input type="password" id="pw-old" required></div>
          <div class="form-group"><label>New Password</label><input type="password" id="pw-new" required></div>
          <div class="form-group"><label>Confirm New Password</label><input type="password" id="pw-confirm" required></div>
          <button type="button" class="btn btn-primary" id="pw-save">Change Password</button>
        </form>
      </div>
    `;
    $("#pw-save").addEventListener("click", async () => {
      const oldPw = $("#pw-old").value;
      const newPw = $("#pw-new").value;
      const confirm2 = $("#pw-confirm").value;
      if (newPw !== confirm2) { toast("Passwords don't match", "error"); return; }
      if (newPw.length < 4) { toast("Password too short", "error"); return; }
      try {
        await api("POST", "/auth/change-password", { old_password: oldPw, new_password: newPw });
        toast("Password changed successfully");
        $("#pw-old").value = ""; $("#pw-new").value = ""; $("#pw-confirm").value = "";
        defaultPwWarn.classList.add("hidden");
      } catch (err) { toast(err.message, "error"); }
    });
  }

  async function loadUsers() {
    try {
      const d = await api("GET", "/api/users");
      const container = $("#tab-users");
      container.innerHTML = `
        <div class="flex-between mb-1">
          <h3>User Management</h3>
          <button class="btn btn-primary btn-sm" id="add-user-btn">+ New User</button>
        </div>
        <div class="table-wrapper"><table>
          <thead><tr><th>Username</th><th>Full Name</th><th>Role</th><th>Created</th><th>Actions</th></tr></thead>
          <tbody>${d.items.map((u) => `<tr>
            <td>${escHtml(u.username)}</td><td>${escHtml(u.full_name)}</td><td>${statusBadge(u.role)}</td>
            <td>${fmtDate(u.created_at)}</td>
            <td>${u.id !== currentUser?.id ? `<button class="btn btn-sm btn-danger del-user" data-id="${u.id}">Delete</button>` : ""}</td>
          </tr>`).join("")}</tbody>
        </table></div>
      `;
      $("#add-user-btn").addEventListener("click", openUserForm);
      container.querySelectorAll(".del-user").forEach((b) =>
        b.addEventListener("click", async () => {
          if (await confirmDialog("Delete User", "Are you sure?")) {
            try { await api("DELETE", `/api/users/${b.dataset.id}`); toast("User deleted"); loadUsers(); }
            catch (err) { toast(err.message, "error"); }
          }
        })
      );
    } catch (err) { toast(err.message, "error"); }
  }

  function openUserForm() {
    const body = `<form id="user-form">
      <div class="form-group"><label>Username</label><input id="uf-username" required></div>
      <div class="form-group"><label>Full Name</label><input id="uf-fullname"></div>
      <div class="form-group"><label>Password</label><input type="password" id="uf-password" required></div>
      <div class="form-group"><label>Role</label><select id="uf-role"><option value="user">User</option><option value="admin">Admin</option></select></div>
    </form>`;
    const footer = `<button class="btn btn-secondary" id="uf-cancel">Cancel</button><button class="btn btn-primary" id="uf-save">Create</button>`;
    openModal("New User", body, footer);
    $("#uf-cancel").addEventListener("click", closeModal);
    $("#uf-save").addEventListener("click", async () => {
      const data = {
        username: $("#uf-username").value.trim(),
        full_name: $("#uf-fullname").value.trim(),
        password: $("#uf-password").value,
        role: $("#uf-role").value,
      };
      if (!data.username || !data.password) { toast("Username and password required", "error"); return; }
      try { await api("POST", "/api/users", data); toast("User created"); closeModal(); loadUsers(); }
      catch (err) { toast(err.message, "error"); }
    });
  }

  // ─── Utility: debounce ──────────────────────────────────────
  function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  // ─── Boot ───────────────────────────────────────────────────
  if (token) {
    initApp();
  }
})();
