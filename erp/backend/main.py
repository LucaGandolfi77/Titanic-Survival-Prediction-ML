# ═══════════════════════════════════════════════════════════════
# ERP System — FastAPI Backend
# ═══════════════════════════════════════════════════════════════
# Startup:
#   cd backend
#   pip install -r requirements.txt
#   python main.py
#   # then open ../index.html in browser
# ═══════════════════════════════════════════════════════════════

import os
import secrets
import math
from datetime import date, datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, DateTime, Date,
    Text, ForeignKey, and_, or_, func, inspect as sa_inspect
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from passlib.context import CryptContext
import jwt
import uvicorn

# ─── Configuration ─────────────────────────────────────────────

DATABASE_URL = "sqlite:///erp.db"
SECRET_KEY = os.environ.get("ERP_SECRET_KEY", "erp-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


def get_db():
    """Yield a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── SQLAlchemy Models ─────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), default="")
    role = Column(String(20), default="user")  # admin, user
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(50), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    category = Column(String(100), default="")
    quantity = Column(Integer, default=0)
    unit_price = Column(Integer, default=0)  # cents
    reorder_level = Column(Integer, default=10)
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    stock_movements = relationship("StockMovement", back_populates="product")
    suppliers = relationship("ProductSupplier", back_populates="product")


class StockMovement(Base):
    __tablename__ = "stock_movements"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    movement_type = Column(String(10), nullable=False)  # IN, OUT
    quantity = Column(Integer, nullable=False)
    reference = Column(String(100), default="")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    product = relationship("Product", back_populates="stock_movements")


class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), default="")
    phone = Column(String(50), default="")
    address = Column(Text, default="")
    company = Column(String(200), default="")
    status = Column(String(20), default="active")  # active, inactive
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sales_orders = relationship("SalesOrder", back_populates="customer")


class Supplier(Base):
    __tablename__ = "suppliers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    contact_person = Column(String(200), default="")
    email = Column(String(200), default="")
    phone = Column(String(50), default="")
    address = Column(Text, default="")
    payment_terms = Column(String(100), default="Net 30")
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    products = relationship("ProductSupplier", back_populates="supplier")
    purchase_orders = relationship("PurchaseOrder", back_populates="supplier")


class ProductSupplier(Base):
    __tablename__ = "product_suppliers"
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    product = relationship("Product", back_populates="suppliers")
    supplier = relationship("Supplier", back_populates="products")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(50), unique=True, nullable=False)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=False)
    status = Column(String(20), default="DRAFT")  # DRAFT, SENT, RECEIVED, CANCELLED
    total_amount = Column(Integer, default=0)  # cents
    notes = Column(Text, default="")
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    supplier = relationship("Supplier", back_populates="purchase_orders")
    items = relationship("PurchaseOrderItem", back_populates="purchase_order", cascade="all, delete-orphan")


class PurchaseOrderItem(Base):
    __tablename__ = "purchase_order_items"
    id = Column(Integer, primary_key=True, index=True)
    purchase_order_id = Column(Integer, ForeignKey("purchase_orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Integer, default=0)  # cents
    purchase_order = relationship("PurchaseOrder", back_populates="items")
    product = relationship("Product")


class SalesOrder(Base):
    __tablename__ = "sales_orders"
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(50), unique=True, nullable=False)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    status = Column(String(20), default="QUOTE")  # QUOTE, CONFIRMED, SHIPPED, INVOICED, CANCELLED
    subtotal = Column(Integer, default=0)  # cents
    tax_rate = Column(Integer, default=22)  # percentage e.g. 22 = 22%
    tax_amount = Column(Integer, default=0)  # cents
    total_amount = Column(Integer, default=0)  # cents
    notes = Column(Text, default="")
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    customer = relationship("Customer", back_populates="sales_orders")
    items = relationship("SalesOrderItem", back_populates="sales_order", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="sales_order")


class SalesOrderItem(Base):
    __tablename__ = "sales_order_items"
    id = Column(Integer, primary_key=True, index=True)
    sales_order_id = Column(Integer, ForeignKey("sales_orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Integer, default=0)  # cents
    sales_order = relationship("SalesOrder", back_populates="items")
    product = relationship("Product")


class Invoice(Base):
    __tablename__ = "invoices"
    id = Column(Integer, primary_key=True, index=True)
    sales_order_id = Column(Integer, ForeignKey("sales_orders.id"), nullable=True)
    invoice_number = Column(String(50), unique=True, nullable=False)
    status = Column(String(20), default="DRAFT")  # DRAFT, SENT, PAID, OVERDUE
    due_date = Column(Date, nullable=True)
    subtotal = Column(Integer, default=0)
    tax_amount = Column(Integer, default=0)
    total_amount = Column(Integer, default=0)
    notes = Column(Text, default="")
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sales_order = relationship("SalesOrder", back_populates="invoices")
    items = relationship("InvoiceItem", back_populates="invoice", cascade="all, delete-orphan")


class InvoiceItem(Base):
    __tablename__ = "invoice_items"
    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), nullable=False)
    description = Column(String(300), default="")
    quantity = Column(Integer, default=1)
    unit_price = Column(Integer, default=0)
    amount = Column(Integer, default=0)
    invoice = relationship("Invoice", back_populates="items")


class Account(Base):
    __tablename__ = "accounts"
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(20), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    account_type = Column(String(20), nullable=False)  # ASSET, LIABILITY, EQUITY, REVENUE, EXPENSE
    balance = Column(Integer, default=0)  # cents
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class JournalEntry(Base):
    __tablename__ = "journal_entries"
    id = Column(Integer, primary_key=True, index=True)
    entry_date = Column(Date, default=date.today)
    description = Column(String(300), default="")
    debit_account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    credit_account_id = Column(Integer, ForeignKey("accounts.id"), nullable=False)
    amount = Column(Integer, default=0)  # cents
    reference_type = Column(String(50), default="")  # invoice, purchase_order
    reference_id = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    debit_account = relationship("Account", foreign_keys=[debit_account_id])
    credit_account = relationship("Account", foreign_keys=[credit_account_id])


class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), default="")
    role = Column(String(100), default="")
    department = Column(String(100), default="")
    salary = Column(Integer, default=0)  # cents (monthly)
    hire_date = Column(Date, default=date.today)
    status = Column(String(20), default="active")  # active, inactive
    is_deleted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class CompanySetting(Base):
    __tablename__ = "company_settings"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text, default="")


# ─── Helper: model to dict ────────────────────────────────────

def to_dict(obj):
    """Convert an SQLAlchemy model instance to a dictionary."""
    d = {}
    for c in sa_inspect(obj.__class__).mapper.column_attrs:
        v = getattr(obj, c.key)
        if isinstance(v, datetime):
            v = v.isoformat()
        elif isinstance(v, date):
            v = v.isoformat()
        d[c.key] = v
    return d


# ─── Pydantic Schemas ─────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int; username: str; full_name: str; role: str; created_at: Optional[str] = None
    class Config: from_attributes = True

class ProductCreate(BaseModel):
    sku: str; name: str; category: str = ""; quantity: int = 0
    unit_price: int = 0; reorder_level: int = 10

class ProductUpdate(BaseModel):
    sku: Optional[str] = None; name: Optional[str] = None; category: Optional[str] = None
    quantity: Optional[int] = None; unit_price: Optional[int] = None; reorder_level: Optional[int] = None

class CustomerCreate(BaseModel):
    name: str; email: str = ""; phone: str = ""; address: str = ""
    company: str = ""; status: str = "active"

class CustomerUpdate(BaseModel):
    name: Optional[str] = None; email: Optional[str] = None; phone: Optional[str] = None
    address: Optional[str] = None; company: Optional[str] = None; status: Optional[str] = None

class SupplierCreate(BaseModel):
    name: str; contact_person: str = ""; email: str = ""; phone: str = ""
    address: str = ""; payment_terms: str = "Net 30"

class SupplierUpdate(BaseModel):
    name: Optional[str] = None; contact_person: Optional[str] = None; email: Optional[str] = None
    phone: Optional[str] = None; address: Optional[str] = None; payment_terms: Optional[str] = None

class OrderItemCreate(BaseModel):
    product_id: int; quantity: int; unit_price: int

class PurchaseOrderCreate(BaseModel):
    supplier_id: int; notes: str = ""; items: List[OrderItemCreate] = []

class PurchaseOrderUpdate(BaseModel):
    supplier_id: Optional[int] = None; notes: Optional[str] = None
    status: Optional[str] = None; items: Optional[List[OrderItemCreate]] = None

class SalesOrderCreate(BaseModel):
    customer_id: int; tax_rate: int = 22; notes: str = ""
    items: List[OrderItemCreate] = []

class SalesOrderUpdate(BaseModel):
    customer_id: Optional[int] = None; tax_rate: Optional[int] = None
    notes: Optional[str] = None; status: Optional[str] = None
    items: Optional[List[OrderItemCreate]] = None

class InvoiceCreate(BaseModel):
    sales_order_id: int

class InvoiceUpdate(BaseModel):
    status: Optional[str] = None; due_date: Optional[str] = None; notes: Optional[str] = None

class AccountCreate(BaseModel):
    code: str; name: str; account_type: str

class JournalEntryCreate(BaseModel):
    entry_date: str; description: str; debit_account_id: int
    credit_account_id: int; amount: int; reference_type: str = ""; reference_id: int = 0

class EmployeeCreate(BaseModel):
    name: str; email: str = ""; role: str = ""; department: str = ""
    salary: int = 0; hire_date: Optional[str] = None; status: str = "active"

class EmployeeUpdate(BaseModel):
    name: Optional[str] = None; email: Optional[str] = None; role: Optional[str] = None
    department: Optional[str] = None; salary: Optional[int] = None
    hire_date: Optional[str] = None; status: Optional[str] = None

class SettingUpdate(BaseModel):
    key: str; value: str

class UserCreate(BaseModel):
    username: str; password: str; full_name: str = ""; role: str = "user"

class ChangePassword(BaseModel):
    old_password: str; new_password: str

class StatusUpdate(BaseModel):
    status: str


# ─── Auth Helpers ──────────────────────────────────────────────

def create_token(user_id: int, username: str, role: str) -> str:
    """Create a JWT access token."""
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    """Dependency: extract and validate current user from JWT."""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id, User.is_deleted == False).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ─── App Lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables and seed data on startup."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            seed_database(db)
        # Check for overdue invoices
        check_overdue_invoices(db)
    finally:
        db.close()
    yield


app = FastAPI(title="ERP System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Auth Routes ───────────────────────────────────────────────

@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token."""
    user = db.query(User).filter(User.username == req.username, User.is_deleted == False).first()
    if not user or not pwd_context.verify(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user.id, user.username, user.role)
    return {"access_token": token}


@app.get("/auth/me")
def get_me(current_user: User = Depends(get_current_user)):
    """Return current user info."""
    is_default = pwd_context.verify("admin123", current_user.password_hash)
    return {**to_dict(current_user), "is_default_password": is_default}


@app.post("/auth/change-password")
def change_password(req: ChangePassword, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Change current user's password."""
    if not pwd_context.verify(req.old_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    current_user.password_hash = pwd_context.hash(req.new_password)
    db.commit()
    return {"message": "Password changed successfully"}


# ─── Dashboard ─────────────────────────────────────────────────

@app.get("/api/dashboard")
def get_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return dashboard KPI data."""
    # Total revenue (from PAID invoices)
    total_revenue = db.query(func.coalesce(func.sum(Invoice.total_amount), 0)).filter(
        Invoice.status == "PAID", Invoice.is_deleted == False
    ).scalar()

    # Open orders
    open_orders = db.query(func.count(SalesOrder.id)).filter(
        SalesOrder.status.in_(["QUOTE", "CONFIRMED"]), SalesOrder.is_deleted == False
    ).scalar()

    # Low stock alerts
    low_stock = db.query(func.count(Product.id)).filter(
        Product.quantity < Product.reorder_level, Product.is_deleted == False
    ).scalar()

    # Active customers
    active_customers = db.query(func.count(Customer.id)).filter(
        Customer.status == "active", Customer.is_deleted == False
    ).scalar()

    # Revenue by month (last 12 months)
    months = []
    now = datetime.utcnow()
    for i in range(11, -1, -1):
        d = now - timedelta(days=30 * i)
        month_start = d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if i > 0:
            next_d = now - timedelta(days=30 * (i - 1))
            month_end = next_d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            month_end = now + timedelta(days=1)
        rev = db.query(func.coalesce(func.sum(Invoice.total_amount), 0)).filter(
            Invoice.status == "PAID",
            Invoice.is_deleted == False,
            Invoice.created_at >= month_start,
            Invoice.created_at < month_end,
        ).scalar()
        months.append({"month": month_start.strftime("%Y-%m"), "revenue": rev})

    # Recent activity (last 10 invoices and orders)
    recent_invoices = db.query(Invoice).filter(Invoice.is_deleted == False).order_by(Invoice.created_at.desc()).limit(5).all()
    recent_sales = db.query(SalesOrder).filter(SalesOrder.is_deleted == False).order_by(SalesOrder.created_at.desc()).limit(5).all()
    activity = []
    for inv in recent_invoices:
        activity.append({"type": "invoice", "description": f"Invoice {inv.invoice_number} — {inv.status}", "date": inv.created_at.isoformat(), "amount": inv.total_amount})
    for so in recent_sales:
        cust = db.query(Customer).get(so.customer_id)
        cname = cust.name if cust else "Unknown"
        activity.append({"type": "sales_order", "description": f"Sales Order {so.order_number} for {cname} — {so.status}", "date": so.created_at.isoformat(), "amount": so.total_amount})
    activity.sort(key=lambda x: x["date"], reverse=True)

    return {
        "total_revenue": total_revenue,
        "open_orders": open_orders,
        "low_stock_alerts": low_stock,
        "active_customers": active_customers,
        "revenue_by_month": months,
        "recent_activity": activity[:10],
    }


# ─── Products ──────────────────────────────────────────────────

@app.get("/api/products")
def list_products(page: int = 1, limit: int = 20, search: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List products with search and pagination."""
    q = db.query(Product).filter(Product.is_deleted == False)
    if search:
        q = q.filter(or_(Product.name.ilike(f"%{search}%"), Product.sku.ilike(f"%{search}%"), Product.category.ilike(f"%{search}%")))
    total = q.count()
    items = q.order_by(Product.id.desc()).offset((page - 1) * limit).limit(limit).all()
    return {"items": [to_dict(i) for i in items], "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/products")
def create_product(data: ProductCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new product."""
    if db.query(Product).filter(Product.sku == data.sku, Product.is_deleted == False).first():
        raise HTTPException(400, "SKU already exists")
    p = Product(**data.model_dump())
    db.add(p)
    db.commit()
    db.refresh(p)
    return to_dict(p)


@app.get("/api/products/{pid}")
def get_product(pid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get product by ID."""
    p = db.query(Product).filter(Product.id == pid, Product.is_deleted == False).first()
    if not p:
        raise HTTPException(404, "Product not found")
    d = to_dict(p)
    d["movements"] = [to_dict(m) for m in db.query(StockMovement).filter(StockMovement.product_id == pid).order_by(StockMovement.created_at.desc()).limit(50).all()]
    d["supplier_ids"] = [ps.supplier_id for ps in db.query(ProductSupplier).filter(ProductSupplier.product_id == pid).all()]
    return d


@app.put("/api/products/{pid}")
def update_product(pid: int, data: ProductUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a product."""
    p = db.query(Product).filter(Product.id == pid, Product.is_deleted == False).first()
    if not p:
        raise HTTPException(404, "Product not found")
    for k, v in data.model_dump(exclude_none=True).items():
        setattr(p, k, v)
    p.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(p)
    return to_dict(p)


@app.delete("/api/products/{pid}")
def delete_product(pid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete a product."""
    p = db.query(Product).filter(Product.id == pid, Product.is_deleted == False).first()
    if not p:
        raise HTTPException(404, "Product not found")
    p.is_deleted = True
    db.commit()
    return {"message": "Deleted"}


@app.get("/api/stock-movements")
def list_stock_movements(product_id: Optional[int] = None, page: int = 1, limit: int = 50, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List stock movements, optionally filtered by product."""
    q = db.query(StockMovement)
    if product_id:
        q = q.filter(StockMovement.product_id == product_id)
    total = q.count()
    items = q.order_by(StockMovement.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    return {"items": [to_dict(i) for i in items], "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


# ─── Customers ─────────────────────────────────────────────────

@app.get("/api/customers")
def list_customers(page: int = 1, limit: int = 20, search: str = "", status: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List customers with search, filter, and pagination."""
    q = db.query(Customer).filter(Customer.is_deleted == False)
    if search:
        q = q.filter(or_(Customer.name.ilike(f"%{search}%"), Customer.email.ilike(f"%{search}%"), Customer.company.ilike(f"%{search}%")))
    if status:
        q = q.filter(Customer.status == status)
    total = q.count()
    items = q.order_by(Customer.id.desc()).offset((page - 1) * limit).limit(limit).all()
    return {"items": [to_dict(i) for i in items], "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/customers")
def create_customer(data: CustomerCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new customer."""
    c = Customer(**data.model_dump())
    db.add(c)
    db.commit()
    db.refresh(c)
    return to_dict(c)


@app.get("/api/customers/{cid}")
def get_customer(cid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get customer by ID with order history."""
    c = db.query(Customer).filter(Customer.id == cid, Customer.is_deleted == False).first()
    if not c:
        raise HTTPException(404, "Customer not found")
    d = to_dict(c)
    orders = db.query(SalesOrder).filter(SalesOrder.customer_id == cid, SalesOrder.is_deleted == False).order_by(SalesOrder.created_at.desc()).all()
    d["orders"] = [to_dict(o) for o in orders]
    d["total_spent"] = db.query(func.coalesce(func.sum(Invoice.total_amount), 0)).join(SalesOrder).filter(SalesOrder.customer_id == cid, Invoice.status == "PAID", Invoice.is_deleted == False).scalar()
    return d


@app.put("/api/customers/{cid}")
def update_customer(cid: int, data: CustomerUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a customer."""
    c = db.query(Customer).filter(Customer.id == cid, Customer.is_deleted == False).first()
    if not c:
        raise HTTPException(404, "Customer not found")
    for k, v in data.model_dump(exclude_none=True).items():
        setattr(c, k, v)
    db.commit()
    db.refresh(c)
    return to_dict(c)


@app.delete("/api/customers/{cid}")
def delete_customer(cid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete a customer."""
    c = db.query(Customer).filter(Customer.id == cid, Customer.is_deleted == False).first()
    if not c:
        raise HTTPException(404, "Customer not found")
    c.is_deleted = True
    db.commit()
    return {"message": "Deleted"}


# ─── Suppliers ─────────────────────────────────────────────────

@app.get("/api/suppliers")
def list_suppliers(page: int = 1, limit: int = 20, search: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List suppliers with search and pagination."""
    q = db.query(Supplier).filter(Supplier.is_deleted == False)
    if search:
        q = q.filter(or_(Supplier.name.ilike(f"%{search}%"), Supplier.contact_person.ilike(f"%{search}%")))
    total = q.count()
    items = q.order_by(Supplier.id.desc()).offset((page - 1) * limit).limit(limit).all()
    return {"items": [to_dict(i) for i in items], "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/suppliers")
def create_supplier(data: SupplierCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new supplier."""
    s = Supplier(**data.model_dump())
    db.add(s)
    db.commit()
    db.refresh(s)
    return to_dict(s)


@app.get("/api/suppliers/{sid}")
def get_supplier(sid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get supplier by ID."""
    s = db.query(Supplier).filter(Supplier.id == sid, Supplier.is_deleted == False).first()
    if not s:
        raise HTTPException(404, "Supplier not found")
    d = to_dict(s)
    d["product_ids"] = [ps.product_id for ps in db.query(ProductSupplier).filter(ProductSupplier.supplier_id == sid).all()]
    return d


@app.put("/api/suppliers/{sid}")
def update_supplier(sid: int, data: SupplierUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a supplier."""
    s = db.query(Supplier).filter(Supplier.id == sid, Supplier.is_deleted == False).first()
    if not s:
        raise HTTPException(404, "Supplier not found")
    for k, v in data.model_dump(exclude_none=True).items():
        setattr(s, k, v)
    db.commit()
    db.refresh(s)
    return to_dict(s)


@app.delete("/api/suppliers/{sid}")
def delete_supplier(sid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete a supplier."""
    s = db.query(Supplier).filter(Supplier.id == sid, Supplier.is_deleted == False).first()
    if not s:
        raise HTTPException(404, "Supplier not found")
    s.is_deleted = True
    db.commit()
    return {"message": "Deleted"}


# ─── Purchase Orders ──────────────────────────────────────────

def next_po_number(db: Session) -> str:
    last = db.query(func.max(PurchaseOrder.id)).scalar() or 0
    return f"PO-{last + 1:04d}"


@app.get("/api/purchase-orders")
def list_purchase_orders(page: int = 1, limit: int = 20, search: str = "", status: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List purchase orders with search and pagination."""
    q = db.query(PurchaseOrder).filter(PurchaseOrder.is_deleted == False)
    if search:
        q = q.filter(PurchaseOrder.order_number.ilike(f"%{search}%"))
    if status:
        q = q.filter(PurchaseOrder.status == status)
    total = q.count()
    items = q.order_by(PurchaseOrder.id.desc()).offset((page - 1) * limit).limit(limit).all()
    result = []
    for po in items:
        d = to_dict(po)
        sup = db.query(Supplier).get(po.supplier_id)
        d["supplier_name"] = sup.name if sup else "Unknown"
        d["item_count"] = len(po.items)
        result.append(d)
    return {"items": result, "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/purchase-orders")
def create_purchase_order(data: PurchaseOrderCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new purchase order with line items."""
    po = PurchaseOrder(order_number=next_po_number(db), supplier_id=data.supplier_id, notes=data.notes)
    db.add(po)
    db.flush()
    total = 0
    for item in data.items:
        poi = PurchaseOrderItem(purchase_order_id=po.id, product_id=item.product_id, quantity=item.quantity, unit_price=item.unit_price)
        db.add(poi)
        total += item.quantity * item.unit_price
    po.total_amount = total
    db.commit()
    db.refresh(po)
    return to_dict(po)


@app.get("/api/purchase-orders/{poid}")
def get_purchase_order(poid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get purchase order by ID with items."""
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == poid, PurchaseOrder.is_deleted == False).first()
    if not po:
        raise HTTPException(404, "Purchase order not found")
    d = to_dict(po)
    sup = db.query(Supplier).get(po.supplier_id)
    d["supplier_name"] = sup.name if sup else "Unknown"
    items = []
    for item in po.items:
        id_dict = to_dict(item)
        prod = db.query(Product).get(item.product_id)
        id_dict["product_name"] = prod.name if prod else "Unknown"
        id_dict["product_sku"] = prod.sku if prod else ""
        items.append(id_dict)
    d["items"] = items
    return d


@app.put("/api/purchase-orders/{poid}")
def update_purchase_order(poid: int, data: PurchaseOrderUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a purchase order."""
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == poid, PurchaseOrder.is_deleted == False).first()
    if not po:
        raise HTTPException(404, "Purchase order not found")
    if data.supplier_id is not None:
        po.supplier_id = data.supplier_id
    if data.notes is not None:
        po.notes = data.notes
    if data.items is not None:
        db.query(PurchaseOrderItem).filter(PurchaseOrderItem.purchase_order_id == poid).delete()
        total = 0
        for item in data.items:
            poi = PurchaseOrderItem(purchase_order_id=poid, product_id=item.product_id, quantity=item.quantity, unit_price=item.unit_price)
            db.add(poi)
            total += item.quantity * item.unit_price
        po.total_amount = total
    po.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(po)
    return to_dict(po)


@app.patch("/api/purchase-orders/{poid}")
def patch_purchase_order(poid: int, data: StatusUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update purchase order status. Auto-update inventory when RECEIVED."""
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == poid, PurchaseOrder.is_deleted == False).first()
    if not po:
        raise HTTPException(404, "Purchase order not found")
    old_status = po.status
    po.status = data.status
    po.updated_at = datetime.utcnow()
    # Auto-update inventory when received
    if data.status == "RECEIVED" and old_status != "RECEIVED":
        for item in po.items:
            prod = db.query(Product).get(item.product_id)
            if prod:
                prod.quantity += item.quantity
                prod.updated_at = datetime.utcnow()
                sm = StockMovement(product_id=prod.id, movement_type="IN", quantity=item.quantity, reference=f"PO-{po.order_number}", notes=f"Purchase order {po.order_number} received")
                db.add(sm)
        # Create journal entry: Debit Inventory, Credit Accounts Payable
        inv_acc = db.query(Account).filter(Account.code == "1200").first()
        ap_acc = db.query(Account).filter(Account.code == "2100").first()
        if inv_acc and ap_acc:
            je = JournalEntry(entry_date=date.today(), description=f"PO {po.order_number} received",
                              debit_account_id=inv_acc.id, credit_account_id=ap_acc.id,
                              amount=po.total_amount, reference_type="purchase_order", reference_id=po.id)
            db.add(je)
            inv_acc.balance += po.total_amount
            ap_acc.balance += po.total_amount
    db.commit()
    return to_dict(po)


@app.delete("/api/purchase-orders/{poid}")
def delete_purchase_order(poid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete a purchase order."""
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == poid, PurchaseOrder.is_deleted == False).first()
    if not po:
        raise HTTPException(404, "Purchase order not found")
    po.is_deleted = True
    db.commit()
    return {"message": "Deleted"}


# ─── Sales Orders ──────────────────────────────────────────────

def next_so_number(db: Session) -> str:
    last = db.query(func.max(SalesOrder.id)).scalar() or 0
    return f"SO-{last + 1:04d}"


@app.get("/api/sales-orders")
def list_sales_orders(page: int = 1, limit: int = 20, search: str = "", status: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List sales orders with search and pagination."""
    q = db.query(SalesOrder).filter(SalesOrder.is_deleted == False)
    if search:
        q = q.filter(SalesOrder.order_number.ilike(f"%{search}%"))
    if status:
        q = q.filter(SalesOrder.status == status)
    total = q.count()
    items = q.order_by(SalesOrder.id.desc()).offset((page - 1) * limit).limit(limit).all()
    result = []
    for so in items:
        d = to_dict(so)
        cust = db.query(Customer).get(so.customer_id)
        d["customer_name"] = cust.name if cust else "Unknown"
        d["item_count"] = len(so.items)
        result.append(d)
    return {"items": result, "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/sales-orders")
def create_sales_order(data: SalesOrderCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new sales order with line items."""
    so = SalesOrder(order_number=next_so_number(db), customer_id=data.customer_id, tax_rate=data.tax_rate, notes=data.notes)
    db.add(so)
    db.flush()
    subtotal = 0
    for item in data.items:
        soi = SalesOrderItem(sales_order_id=so.id, product_id=item.product_id, quantity=item.quantity, unit_price=item.unit_price)
        db.add(soi)
        subtotal += item.quantity * item.unit_price
    so.subtotal = subtotal
    so.tax_amount = round(subtotal * so.tax_rate / 100)
    so.total_amount = subtotal + so.tax_amount
    db.commit()
    db.refresh(so)
    return to_dict(so)


@app.get("/api/sales-orders/{soid}")
def get_sales_order(soid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get sales order by ID with items."""
    so = db.query(SalesOrder).filter(SalesOrder.id == soid, SalesOrder.is_deleted == False).first()
    if not so:
        raise HTTPException(404, "Sales order not found")
    d = to_dict(so)
    cust = db.query(Customer).get(so.customer_id)
    d["customer_name"] = cust.name if cust else "Unknown"
    items = []
    for item in so.items:
        id_dict = to_dict(item)
        prod = db.query(Product).get(item.product_id)
        id_dict["product_name"] = prod.name if prod else "Unknown"
        id_dict["product_sku"] = prod.sku if prod else ""
        items.append(id_dict)
    d["items"] = items
    return d


@app.put("/api/sales-orders/{soid}")
def update_sales_order(soid: int, data: SalesOrderUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update a sales order."""
    so = db.query(SalesOrder).filter(SalesOrder.id == soid, SalesOrder.is_deleted == False).first()
    if not so:
        raise HTTPException(404, "Sales order not found")
    if data.customer_id is not None:
        so.customer_id = data.customer_id
    if data.tax_rate is not None:
        so.tax_rate = data.tax_rate
    if data.notes is not None:
        so.notes = data.notes
    if data.items is not None:
        db.query(SalesOrderItem).filter(SalesOrderItem.sales_order_id == soid).delete()
        subtotal = 0
        for item in data.items:
            soi = SalesOrderItem(sales_order_id=soid, product_id=item.product_id, quantity=item.quantity, unit_price=item.unit_price)
            db.add(soi)
            subtotal += item.quantity * item.unit_price
        so.subtotal = subtotal
        so.tax_amount = round(subtotal * so.tax_rate / 100)
        so.total_amount = subtotal + so.tax_amount
    so.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(so)
    return to_dict(so)


@app.patch("/api/sales-orders/{soid}")
def patch_sales_order(soid: int, data: StatusUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update sales order status. Auto-deduct inventory when SHIPPED."""
    so = db.query(SalesOrder).filter(SalesOrder.id == soid, SalesOrder.is_deleted == False).first()
    if not so:
        raise HTTPException(404, "Sales order not found")
    old_status = so.status
    so.status = data.status
    so.updated_at = datetime.utcnow()
    # Auto-deduct inventory when shipped
    if data.status == "SHIPPED" and old_status != "SHIPPED":
        for item in so.items:
            prod = db.query(Product).get(item.product_id)
            if prod:
                prod.quantity = max(0, prod.quantity - item.quantity)
                prod.updated_at = datetime.utcnow()
                sm = StockMovement(product_id=prod.id, movement_type="OUT", quantity=item.quantity, reference=f"SO-{so.order_number}", notes=f"Sales order {so.order_number} shipped")
                db.add(sm)
    db.commit()
    return to_dict(so)


@app.delete("/api/sales-orders/{soid}")
def delete_sales_order(soid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete a sales order."""
    so = db.query(SalesOrder).filter(SalesOrder.id == soid, SalesOrder.is_deleted == False).first()
    if not so:
        raise HTTPException(404, "Sales order not found")
    so.is_deleted = True
    db.commit()
    return {"message": "Deleted"}


# ─── Invoices ──────────────────────────────────────────────────

def next_invoice_number(db: Session) -> str:
    last = db.query(func.max(Invoice.id)).scalar() or 0
    return f"INV-{last + 1:04d}"


def check_overdue_invoices(db: Session):
    """Mark SENT invoices past due_date as OVERDUE."""
    today = date.today()
    overdue = db.query(Invoice).filter(
        Invoice.status == "SENT",
        Invoice.due_date < today,
        Invoice.is_deleted == False,
    ).all()
    for inv in overdue:
        inv.status = "OVERDUE"
    if overdue:
        db.commit()


@app.get("/api/invoices")
def list_invoices(page: int = 1, limit: int = 20, search: str = "", status: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List invoices with search and pagination."""
    check_overdue_invoices(db)
    q = db.query(Invoice).filter(Invoice.is_deleted == False)
    if search:
        q = q.filter(Invoice.invoice_number.ilike(f"%{search}%"))
    if status:
        q = q.filter(Invoice.status == status)
    total = q.count()
    items = q.order_by(Invoice.id.desc()).offset((page - 1) * limit).limit(limit).all()
    result = []
    for inv in items:
        d = to_dict(inv)
        if inv.sales_order_id:
            so = db.query(SalesOrder).get(inv.sales_order_id)
            if so:
                cust = db.query(Customer).get(so.customer_id)
                d["customer_name"] = cust.name if cust else "Unknown"
                d["order_number"] = so.order_number
        result.append(d)
    return {"items": result, "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/invoices")
def create_invoice(data: InvoiceCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Auto-generate invoice from a sales order."""
    so = db.query(SalesOrder).filter(SalesOrder.id == data.sales_order_id, SalesOrder.is_deleted == False).first()
    if not so:
        raise HTTPException(404, "Sales order not found")
    inv = Invoice(
        sales_order_id=so.id, invoice_number=next_invoice_number(db),
        subtotal=so.subtotal, tax_amount=so.tax_amount, total_amount=so.total_amount,
        due_date=date.today() + timedelta(days=30),
    )
    db.add(inv)
    db.flush()
    # Copy line items
    for soi in so.items:
        prod = db.query(Product).get(soi.product_id)
        ii = InvoiceItem(
            invoice_id=inv.id,
            description=prod.name if prod else f"Product #{soi.product_id}",
            quantity=soi.quantity, unit_price=soi.unit_price,
            amount=soi.quantity * soi.unit_price,
        )
        db.add(ii)
    # Update SO status
    so.status = "INVOICED"
    so.updated_at = datetime.utcnow()
    # Create journal entry: Debit Accounts Receivable, Credit Revenue
    ar_acc = db.query(Account).filter(Account.code == "1300").first()
    rev_acc = db.query(Account).filter(Account.code == "4000").first()
    if ar_acc and rev_acc:
        je = JournalEntry(entry_date=date.today(), description=f"Invoice {inv.invoice_number} issued",
                          debit_account_id=ar_acc.id, credit_account_id=rev_acc.id,
                          amount=inv.total_amount, reference_type="invoice", reference_id=inv.id)
        db.add(je)
        ar_acc.balance += inv.total_amount
        rev_acc.balance += inv.total_amount
    db.commit()
    db.refresh(inv)
    return to_dict(inv)


@app.get("/api/invoices/{iid}")
def get_invoice(iid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get invoice by ID with items."""
    inv = db.query(Invoice).filter(Invoice.id == iid, Invoice.is_deleted == False).first()
    if not inv:
        raise HTTPException(404, "Invoice not found")
    d = to_dict(inv)
    d["items"] = [to_dict(ii) for ii in inv.items]
    if inv.sales_order_id:
        so = db.query(SalesOrder).get(inv.sales_order_id)
        if so:
            cust = db.query(Customer).get(so.customer_id)
            d["customer_name"] = cust.name if cust else "Unknown"
            d["order_number"] = so.order_number
    return d


@app.patch("/api/invoices/{iid}")
def patch_invoice(iid: int, data: InvoiceUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update invoice status or details."""
    inv = db.query(Invoice).filter(Invoice.id == iid, Invoice.is_deleted == False).first()
    if not inv:
        raise HTTPException(404, "Invoice not found")
    old_status = inv.status
    if data.status:
        inv.status = data.status
    if data.due_date:
        inv.due_date = date.fromisoformat(data.due_date)
    if data.notes is not None:
        inv.notes = data.notes
    inv.updated_at = datetime.utcnow()
    # When paid, create journal entry: Debit Cash, Credit Accounts Receivable
    if data.status == "PAID" and old_status != "PAID":
        cash_acc = db.query(Account).filter(Account.code == "1100").first()
        ar_acc = db.query(Account).filter(Account.code == "1300").first()
        if cash_acc and ar_acc:
            je = JournalEntry(entry_date=date.today(), description=f"Payment received for {inv.invoice_number}",
                              debit_account_id=cash_acc.id, credit_account_id=ar_acc.id,
                              amount=inv.total_amount, reference_type="invoice", reference_id=inv.id)
            db.add(je)
            cash_acc.balance += inv.total_amount
            ar_acc.balance -= inv.total_amount
    db.commit()
    db.refresh(inv)
    return to_dict(inv)


@app.delete("/api/invoices/{iid}")
def delete_invoice(iid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete an invoice."""
    inv = db.query(Invoice).filter(Invoice.id == iid, Invoice.is_deleted == False).first()
    if not inv:
        raise HTTPException(404, "Invoice not found")
    inv.is_deleted = True
    db.commit()
    return {"message": "Deleted"}


# ─── Accounting ────────────────────────────────────────────────

@app.get("/api/accounts")
def list_accounts(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all accounts (chart of accounts)."""
    items = db.query(Account).filter(Account.is_deleted == False).order_by(Account.code).all()
    return {"items": [to_dict(a) for a in items]}


@app.post("/api/accounts")
def create_account(data: AccountCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new account."""
    if db.query(Account).filter(Account.code == data.code).first():
        raise HTTPException(400, "Account code already exists")
    a = Account(**data.model_dump())
    db.add(a)
    db.commit()
    db.refresh(a)
    return to_dict(a)


@app.get("/api/journal-entries")
def list_journal_entries(page: int = 1, limit: int = 50, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List journal entries."""
    q = db.query(JournalEntry)
    total = q.count()
    items = q.order_by(JournalEntry.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    result = []
    for je in items:
        d = to_dict(je)
        d["debit_account_name"] = je.debit_account.name if je.debit_account else ""
        d["credit_account_name"] = je.credit_account.name if je.credit_account else ""
        result.append(d)
    return {"items": result, "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/journal-entries")
def create_journal_entry(data: JournalEntryCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a manual journal entry."""
    je = JournalEntry(
        entry_date=date.fromisoformat(data.entry_date), description=data.description,
        debit_account_id=data.debit_account_id, credit_account_id=data.credit_account_id,
        amount=data.amount, reference_type=data.reference_type, reference_id=data.reference_id,
    )
    db.add(je)
    # Update account balances
    debit_acc = db.query(Account).get(data.debit_account_id)
    credit_acc = db.query(Account).get(data.credit_account_id)
    if debit_acc:
        debit_acc.balance += data.amount
    if credit_acc:
        credit_acc.balance += data.amount
    db.commit()
    db.refresh(je)
    return to_dict(je)


@app.get("/api/balance-sheet")
def get_balance_sheet(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get balance sheet summary."""
    accounts = db.query(Account).filter(Account.is_deleted == False).all()
    assets = [to_dict(a) for a in accounts if a.account_type == "ASSET"]
    liabilities = [to_dict(a) for a in accounts if a.account_type == "LIABILITY"]
    equity = [to_dict(a) for a in accounts if a.account_type == "EQUITY"]
    total_assets = sum(a["balance"] for a in assets)
    total_liabilities = sum(a["balance"] for a in liabilities)
    total_equity = sum(a["balance"] for a in equity)
    return {
        "assets": assets, "liabilities": liabilities, "equity": equity,
        "total_assets": total_assets, "total_liabilities": total_liabilities, "total_equity": total_equity,
    }


@app.get("/api/profit-loss")
def get_profit_loss(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get Profit & Loss summary by month."""
    accounts = db.query(Account).filter(Account.is_deleted == False).all()
    revenue_accs = [a for a in accounts if a.account_type == "REVENUE"]
    expense_accs = [a for a in accounts if a.account_type == "EXPENSE"]
    total_revenue = sum(a.balance for a in revenue_accs)
    total_expenses = sum(a.balance for a in expense_accs)
    # Monthly breakdown from journal entries for the current year
    year = date.today().year
    months = []
    for m in range(1, 13):
        start = datetime(year, m, 1)
        end = datetime(year, m + 1, 1) if m < 12 else datetime(year + 1, 1, 1)
        rev = db.query(func.coalesce(func.sum(JournalEntry.amount), 0)).filter(
            JournalEntry.credit_account_id.in_([a.id for a in revenue_accs]),
            JournalEntry.entry_date >= start.date(), JournalEntry.entry_date < end.date(),
        ).scalar()
        exp = db.query(func.coalesce(func.sum(JournalEntry.amount), 0)).filter(
            JournalEntry.debit_account_id.in_([a.id for a in expense_accs]),
            JournalEntry.entry_date >= start.date(), JournalEntry.entry_date < end.date(),
        ).scalar()
        months.append({"month": f"{year}-{m:02d}", "revenue": rev, "expenses": exp, "profit": rev - exp})
    return {
        "revenue_accounts": [to_dict(a) for a in revenue_accs],
        "expense_accounts": [to_dict(a) for a in expense_accs],
        "total_revenue": total_revenue, "total_expenses": total_expenses,
        "net_profit": total_revenue - total_expenses,
        "monthly": months,
    }


# ─── Employees ─────────────────────────────────────────────────

@app.get("/api/employees")
def list_employees(page: int = 1, limit: int = 20, search: str = "", department: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List employees."""
    q = db.query(Employee).filter(Employee.is_deleted == False)
    if search:
        q = q.filter(or_(Employee.name.ilike(f"%{search}%"), Employee.role.ilike(f"%{search}%")))
    if department:
        q = q.filter(Employee.department == department)
    total = q.count()
    items = q.order_by(Employee.id.desc()).offset((page - 1) * limit).limit(limit).all()
    return {"items": [to_dict(e) for e in items], "total": total, "page": page, "pages": math.ceil(total / limit) if limit else 1}


@app.post("/api/employees")
def create_employee(data: EmployeeCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new employee."""
    d = data.model_dump()
    if d.get("hire_date"):
        d["hire_date"] = date.fromisoformat(d["hire_date"])
    else:
        d["hire_date"] = date.today()
    e = Employee(**d)
    db.add(e)
    db.commit()
    db.refresh(e)
    return to_dict(e)


@app.get("/api/employees/{eid}")
def get_employee(eid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get employee by ID."""
    e = db.query(Employee).filter(Employee.id == eid, Employee.is_deleted == False).first()
    if not e:
        raise HTTPException(404, "Employee not found")
    return to_dict(e)


@app.put("/api/employees/{eid}")
def update_employee(eid: int, data: EmployeeUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update an employee."""
    e = db.query(Employee).filter(Employee.id == eid, Employee.is_deleted == False).first()
    if not e:
        raise HTTPException(404, "Employee not found")
    for k, v in data.model_dump(exclude_none=True).items():
        if k == "hire_date" and v:
            v = date.fromisoformat(v)
        setattr(e, k, v)
    db.commit()
    db.refresh(e)
    return to_dict(e)


@app.delete("/api/employees/{eid}")
def delete_employee(eid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete (deactivate) an employee."""
    e = db.query(Employee).filter(Employee.id == eid, Employee.is_deleted == False).first()
    if not e:
        raise HTTPException(404, "Employee not found")
    e.is_deleted = True
    e.status = "inactive"
    db.commit()
    return {"message": "Deleted"}


@app.get("/api/payroll-summary")
def payroll_summary(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get monthly payroll summary by department."""
    results = db.query(Employee.department, func.sum(Employee.salary), func.count(Employee.id)).filter(
        Employee.is_deleted == False, Employee.status == "active"
    ).group_by(Employee.department).all()
    departments = [{"department": r[0] or "Unassigned", "total_salary": r[1], "employee_count": r[2]} for r in results]
    grand_total = sum(d["total_salary"] for d in departments)
    return {"departments": departments, "grand_total": grand_total}


# ─── Settings ──────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all company settings."""
    settings = db.query(CompanySetting).all()
    return {s.key: s.value for s in settings}


@app.put("/api/settings")
def update_settings(data: List[SettingUpdate], current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update company settings (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    for item in data:
        existing = db.query(CompanySetting).filter(CompanySetting.key == item.key).first()
        if existing:
            existing.value = item.value
        else:
            db.add(CompanySetting(key=item.key, value=item.value))
    db.commit()
    return {"message": "Settings updated"}


# ─── Users Management (admin only) ────────────────────────────

@app.get("/api/users")
def list_users(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List all users (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    users = db.query(User).filter(User.is_deleted == False).all()
    return {"items": [{**to_dict(u), "password_hash": None} for u in users]}


@app.post("/api/users")
def create_user(data: UserCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Create a new user (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    if db.query(User).filter(User.username == data.username, User.is_deleted == False).first():
        raise HTTPException(400, "Username already exists")
    u = User(username=data.username, password_hash=pwd_context.hash(data.password), full_name=data.full_name, role=data.role)
    db.add(u)
    db.commit()
    db.refresh(u)
    return {**to_dict(u), "password_hash": None}


@app.delete("/api/users/{uid}")
def delete_user(uid: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Soft-delete a user (admin only)."""
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")
    if uid == current_user.id:
        raise HTTPException(400, "Cannot delete yourself")
    u = db.query(User).filter(User.id == uid, User.is_deleted == False).first()
    if not u:
        raise HTTPException(404, "User not found")
    u.is_deleted = True
    db.commit()
    return {"message": "Deleted"}


# ─── Seed Database ─────────────────────────────────────────────

def seed_database(db: Session):
    """Populate database with sample data on first run."""
    import random
    random.seed(42)

    # Admin user
    admin = User(username="admin", password_hash=pwd_context.hash("admin123"), full_name="System Administrator", role="admin")
    db.add(admin)
    user2 = User(username="manager", password_hash=pwd_context.hash("manager123"), full_name="Jane Manager", role="user")
    db.add(user2)
    db.flush()

    # Chart of Accounts
    accounts_data = [
        ("1000", "Cash on Hand", "ASSET"),
        ("1100", "Bank Account", "ASSET"),
        ("1200", "Inventory", "ASSET"),
        ("1300", "Accounts Receivable", "ASSET"),
        ("2000", "Accounts Payable", "LIABILITY"),
        ("2100", "Supplier Payables", "LIABILITY"),
        ("2200", "Tax Payable", "LIABILITY"),
        ("3000", "Owner's Equity", "EQUITY"),
        ("3100", "Retained Earnings", "EQUITY"),
        ("4000", "Sales Revenue", "REVENUE"),
        ("4100", "Service Revenue", "REVENUE"),
        ("5000", "Cost of Goods Sold", "EXPENSE"),
        ("5100", "Salaries Expense", "EXPENSE"),
        ("5200", "Rent Expense", "EXPENSE"),
        ("5300", "Utilities Expense", "EXPENSE"),
    ]
    accs = {}
    for code, name, atype in accounts_data:
        a = Account(code=code, name=name, account_type=atype, balance=0)
        db.add(a)
        db.flush()
        accs[code] = a

    # Seed initial balance for bank
    accs["1100"].balance = 50000000  # $500,000

    # Company Settings
    settings = [
        ("company_name", "Acme Industries Ltd."),
        ("company_address", "123 Enterprise Avenue, Business City, BC 10001"),
        ("company_vat", "IT12345678901"),
        ("company_logo", ""),
        ("currency", "EUR"),
        ("currency_symbol", "€"),
        ("default_tax_rate", "22"),
    ]
    for k, v in settings:
        db.add(CompanySetting(key=k, value=v))

    # Products (25)
    products_data = [
        ("SKU-001", "Ergonomic Office Chair", "Furniture", 45, 24900, 10),
        ("SKU-002", "Standing Desk Pro", "Furniture", 30, 44900, 8),
        ("SKU-003", "27\" 4K Monitor", "Electronics", 60, 34900, 15),
        ("SKU-004", "Mechanical Keyboard", "Electronics", 100, 8900, 20),
        ("SKU-005", "Wireless Mouse", "Electronics", 150, 3900, 25),
        ("SKU-006", "HD Webcam", "Electronics", 80, 7900, 15),
        ("SKU-007", "Noise-Cancelling Headset", "Electronics", 55, 12900, 10),
        ("SKU-008", "Laptop Stand Adjustable", "Accessories", 70, 4500, 15),
        ("SKU-009", "Cable Management Kit", "Accessories", 200, 1900, 30),
        ("SKU-010", "Whiteboard 120x90cm", "Office", 20, 8900, 5),
        ("SKU-011", "Printer Paper A4 (5000)", "Supplies", 300, 3200, 50),
        ("SKU-012", "Laser Toner Black", "Supplies", 40, 5900, 10),
        ("SKU-013", "USB-C Hub 7-port", "Electronics", 90, 4900, 20),
        ("SKU-014", "Cat6 Ethernet Cable 5m", "Supplies", 500, 890, 100),
        ("SKU-015", "Power Strip 6-outlet", "Accessories", 120, 2490, 25),
        ("SKU-016", "LED Desk Lamp", "Accessories", 65, 3900, 15),
        ("SKU-017", "Filing Cabinet 3-drawer", "Furniture", 25, 15900, 5),
        ("SKU-018", "Paper Shredder", "Office", 15, 12900, 3),
        ("SKU-019", "Stapler Heavy Duty", "Supplies", 100, 1490, 20),
        ("SKU-020", "Notebook Pack (10)", "Supplies", 250, 1990, 40),
        ("SKU-021", "Pen Set Premium (12)", "Supplies", 180, 890, 30),
        ("SKU-022", "Laser Printer All-in-One", "Electronics", 12, 29900, 3),
        ("SKU-023", "Document Scanner", "Electronics", 18, 19900, 5),
        ("SKU-024", "External SSD 1TB", "Electronics", 35, 8900, 8),
        ("SKU-025", "Wireless Router AX6000", "Electronics", 28, 14900, 5),
    ]
    prods = []
    for sku, name, cat, qty, price, reorder in products_data:
        p = Product(sku=sku, name=name, category=cat, quantity=qty, unit_price=price, reorder_level=reorder)
        db.add(p)
        db.flush()
        prods.append(p)
        # Initial stock movement
        db.add(StockMovement(product_id=p.id, movement_type="IN", quantity=qty, reference="INIT", notes="Initial stock"))

    # Customers (22)
    customers_data = [
        ("Acme Corporation", "john@acme.com", "+1-555-0101", "100 Main St, New York, NY", "Acme Corp"),
        ("TechFlow Solutions", "info@techflow.io", "+1-555-0102", "200 Tech Blvd, San Francisco, CA", "TechFlow Inc"),
        ("GreenLeaf Organics", "orders@greenleaf.com", "+1-555-0103", "50 Garden Way, Portland, OR", "GreenLeaf LLC"),
        ("Pinnacle Consulting", "admin@pinnacle.co", "+1-555-0104", "75 Business Park, Chicago, IL", "Pinnacle Group"),
        ("Stellar Dynamics", "contact@stellar.dev", "+1-555-0105", "300 Innovation Dr, Austin, TX", "Stellar Inc"),
        ("NorthStar Logistics", "ops@northstar.com", "+1-555-0106", "80 Harbor Rd, Seattle, WA", "NorthStar LLC"),
        ("BlueOcean Media", "hello@blueocean.co", "+1-555-0107", "45 Media Row, Los Angeles, CA", "BlueOcean Inc"),
        ("IronBridge Construction", "projects@ironbridge.com", "+1-555-0108", "120 Industrial Pkwy, Denver, CO", "IronBridge Ltd"),
        ("Quantum Analytics", "data@quantum-a.com", "+1-555-0109", "500 Science Park, Boston, MA", "Quantum Corp"),
        ("SwiftRelay Systems", "support@swiftrelay.io", "+1-555-0110", "90 Sprint Ave, Dallas, TX", "SwiftRelay Inc"),
        ("HarborView Hotels", "procurement@harborview.com", "+1-555-0111", "1 Waterfront Dr, Miami, FL", "HarborView Group"),
        ("Crestline Education", "admin@crestline.edu", "+1-555-0112", "25 Campus Way, Atlanta, GA", "Crestline Schools"),
        ("Redwood Manufacturing", "supply@redwood-mfg.com", "+1-555-0113", "400 Factory Ln, Detroit, MI", "Redwood Mfg"),
        ("Vanguard Financial", "office@vanguardfi.com", "+1-555-0114", "60 Wall St, New York, NY", "Vanguard Fin"),
        ("Apex Sports Gear", "orders@apexsports.com", "+1-555-0115", "15 Stadium Rd, Phoenix, AZ", "Apex Sports"),
        ("MeridianHealth", "admin@meridianhealth.org", "+1-555-0116", "700 Health Center, Philadelphia, PA", "MeridianHealth"),
        ("Cascade Energy", "info@cascade-energy.com", "+1-555-0117", "85 Power Pl, Portland, OR", "Cascade Energy"),
        ("Summit Legal Group", "clerk@summitlegal.com", "+1-555-0118", "55 Court St, Washington, DC", "Summit Legal"),
        ("PrimeAuto Parts", "warehouse@primeauto.com", "+1-555-0119", "200 Motor Mile, Nashville, TN", "PrimeAuto"),
        ("LunaDesign Studio", "info@lunadesign.co", "+1-555-0120", "10 Creative Ln, San Diego, CA", "LunaDesign"),
        ("Titanium Security", "ops@tisec.com", "+1-555-0121", "30 Shield Rd, Las Vegas, NV", "Titanium Sec"),
        ("FreshBrew Coffee Co.", "orders@freshbrew.co", "+1-555-0122", "5 Roast Ave, Minneapolis, MN", "FreshBrew"),
    ]
    custs = []
    for name, email, phone, addr, company in customers_data:
        c = Customer(name=name, email=email, phone=phone, address=addr, company=company,
                     status=random.choice(["active", "active", "active", "inactive"]))
        db.add(c)
        db.flush()
        custs.append(c)

    # Suppliers (22)
    suppliers_data = [
        ("OfficeMax Direct", "Tom Baker", "tom@officemax-d.com", "+1-555-0201", "10 Supply Chain Rd, Columbus, OH", "Net 30"),
        ("TechParts Global", "Sarah Chen", "sarah@techparts.com", "+1-555-0202", "500 Components Ave, San Jose, CA", "Net 45"),
        ("FurniturePro B2B", "Mike Wilson", "mike@furniturepro.com", "+1-555-0203", "75 Warehouse Blvd, Grand Rapids, MI", "Net 30"),
        ("CableWorld Inc", "Lisa Patel", "lisa@cableworld.com", "+1-555-0204", "20 Connector St, Shenzhen, CN", "Net 60"),
        ("PrintSupply Co", "David Kim", "david@printsupply.com", "+1-555-0205", "35 Toner Ln, Memphis, TN", "Net 30"),
        ("AudioTech Direct", "Emma Brown", "emma@audiotech.com", "+1-555-0206", "60 Sound Way, Nashville, TN", "Net 45"),
        ("ScreenMaster Displays", "James Lee", "james@screenmaster.com", "+1-555-0207", "150 Display Dr, Seoul, KR", "Net 60"),
        ("KeySwitch Peripherals", "Amy Turner", "amy@keyswitch.com", "+1-555-0208", "40 Input Ave, Taipei, TW", "Net 30"),
        ("LightWorks Lighting", "Carlos Diaz", "carlos@lightworks.com", "+1-555-0209", "25 Lumen St, Milwaukee, WI", "Net 30"),
        ("NetGear Solutions", "Rachel Fox", "rachel@netgear-s.com", "+1-555-0210", "80 Network Rd, San Jose, CA", "Net 45"),
        ("StoragePro SSDs", "Kevin Park", "kevin@storagepro.com", "+1-555-0211", "100 Flash Ave, Austin, TX", "Net 30"),
        ("ErgoDirect Supply", "Patricia Hernandez", "pat@ergodirect.com", "+1-555-0212", "55 Comfort Blvd, Charlotte, NC", "Net 30"),
        ("PaperMill Express", "Robert Taylor", "robert@papermill.com", "+1-555-0213", "300 Pulp Rd, Portland, ME", "Net 15"),
        ("SecureShred Corp", "Nancy White", "nancy@secureshred.com", "+1-555-0214", "15 Destroy St, Indianapolis, IN", "Net 30"),
        ("StationeryBulk Ltd", "George Martin", "george@stbulk.com", "+1-555-0215", "45 Pen Ct, Newark, NJ", "Net 30"),
        ("PowerStrip Direct", "Helen Clark", "helen@psdirect.com", "+1-555-0216", "70 Outlet Way, Phoenix, AZ", "Net 45"),
        ("WebcamPro Supply", "Frank Adams", "frank@webcampro.com", "+1-555-0217", "30 Lens Rd, San Francisco, CA", "Net 30"),
        ("ScanTech Solutions", "Diana Evans", "diana@scantech.com", "+1-555-0218", "90 Scanner Pl, Denver, CO", "Net 30"),
        ("HubConnect Parts", "Steven Hill", "steven@hubconnect.com", "+1-555-0219", "22 Port St, Portland, OR", "Net 30"),
        ("WhiteboardWorld", "Marie Johnson", "marie@wbworld.com", "+1-555-0220", "10 Marker Ln, Minneapolis, MN", "Net 15"),
        ("BulkElectronics Intl", "Alan Wright", "alan@bulkelec.com", "+1-555-0221", "400 Chip Ave, Hong Kong, HK", "Net 60"),
        ("GreenOffice Supplies", "Catherine Scott", "cathy@greenoffice.com", "+1-555-0222", "5 Eco Rd, Seattle, WA", "Net 30"),
    ]
    supps = []
    for name, cp, email, phone, addr, terms in suppliers_data:
        s = Supplier(name=name, contact_person=cp, email=email, phone=phone, address=addr, payment_terms=terms)
        db.add(s)
        db.flush()
        supps.append(s)

    # Link some suppliers to products
    links = [(0, 0), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5), (1, 6), (2, 0), (2, 1), (3, 13),
             (4, 11), (5, 6), (6, 2), (7, 3), (7, 4), (8, 15), (9, 24), (10, 23), (11, 0), (11, 7)]
    for si, pi in links:
        if si < len(supps) and pi < len(prods):
            db.add(ProductSupplier(product_id=prods[pi].id, supplier_id=supps[si].id))

    # Employees (22)
    employees_data = [
        ("Alice Johnson", "alice.j@company.com", "CEO", "Executive", 1200000),
        ("Bob Smith", "bob.s@company.com", "CTO", "Executive", 1100000),
        ("Carol Williams", "carol.w@company.com", "CFO", "Finance", 1050000),
        ("David Brown", "david.b@company.com", "Senior Developer", "Engineering", 750000),
        ("Eva Martinez", "eva.m@company.com", "Full-Stack Developer", "Engineering", 680000),
        ("Frank Lee", "frank.l@company.com", "Backend Developer", "Engineering", 650000),
        ("Grace Chen", "grace.c@company.com", "Frontend Developer", "Engineering", 640000),
        ("Henry Wilson", "henry.w@company.com", "DevOps Engineer", "Engineering", 700000),
        ("Iris Patel", "iris.p@company.com", "QA Lead", "Engineering", 620000),
        ("Jack Taylor", "jack.t@company.com", "Sales Director", "Sales", 850000),
        ("Karen Davis", "karen.d@company.com", "Account Executive", "Sales", 550000),
        ("Leo Anderson", "leo.a@company.com", "Account Executive", "Sales", 520000),
        ("Mary Thomas", "mary.t@company.com", "Marketing Manager", "Marketing", 680000),
        ("Nathan Harris", "nathan.h@company.com", "Content Specialist", "Marketing", 450000),
        ("Olivia Robinson", "olivia.r@company.com", "HR Director", "Human Resources", 780000),
        ("Peter Clark", "peter.c@company.com", "HR Coordinator", "Human Resources", 420000),
        ("Quinn Walker", "quinn.w@company.com", "Accountant", "Finance", 520000),
        ("Rachel Green", "rachel.g@company.com", "Financial Analyst", "Finance", 580000),
        ("Sam King", "sam.k@company.com", "Warehouse Manager", "Operations", 480000),
        ("Tina Wright", "tina.w@company.com", "Logistics Coordinator", "Operations", 420000),
        ("Ulrich Fischer", "ulrich.f@company.com", "IT Support", "IT", 450000),
        ("Victoria Santos", "victoria.s@company.com", "Office Manager", "Administration", 460000),
    ]
    for name, email, role, dept, salary in employees_data:
        hire = date.today() - timedelta(days=random.randint(60, 1500))
        db.add(Employee(name=name, email=email, role=role, department=dept, salary=salary, hire_date=hire))
    db.flush()

    # Purchase Orders (22)
    po_statuses = ["RECEIVED", "RECEIVED", "RECEIVED", "RECEIVED", "RECEIVED",
                   "RECEIVED", "RECEIVED", "RECEIVED", "SENT", "SENT",
                   "SENT", "DRAFT", "DRAFT", "RECEIVED", "RECEIVED",
                   "RECEIVED", "SENT", "DRAFT", "RECEIVED", "RECEIVED", "CANCELLED", "RECEIVED"]
    for i in range(22):
        sup = supps[i % len(supps)]
        po = PurchaseOrder(order_number=f"PO-{i+1:04d}", supplier_id=sup.id, status=po_statuses[i],
                           created_at=datetime.utcnow() - timedelta(days=random.randint(5, 300)))
        db.add(po)
        db.flush()
        total = 0
        num_items = random.randint(1, 4)
        used = set()
        for _ in range(num_items):
            p = random.choice(prods)
            if p.id in used:
                continue
            used.add(p.id)
            qty = random.randint(5, 50)
            price = int(p.unit_price * 0.6)  # purchase at 60% of sale price
            poi = PurchaseOrderItem(purchase_order_id=po.id, product_id=p.id, quantity=qty, unit_price=price)
            db.add(poi)
            total += qty * price
        po.total_amount = total
    db.flush()

    # Sales Orders (24) - in various states
    so_statuses = ["INVOICED", "INVOICED", "INVOICED", "INVOICED", "INVOICED",
                   "INVOICED", "INVOICED", "INVOICED", "INVOICED", "INVOICED",
                   "SHIPPED", "SHIPPED", "SHIPPED", "CONFIRMED", "CONFIRMED",
                   "CONFIRMED", "QUOTE", "QUOTE", "QUOTE", "INVOICED",
                   "INVOICED", "INVOICED", "CANCELLED", "INVOICED"]
    sos = []
    for i in range(24):
        cust = custs[i % len(custs)]
        tax_rate = 22
        so = SalesOrder(order_number=f"SO-{i+1:04d}", customer_id=cust.id, status=so_statuses[i],
                        tax_rate=tax_rate, created_at=datetime.utcnow() - timedelta(days=random.randint(5, 350)))
        db.add(so)
        db.flush()
        subtotal = 0
        num_items = random.randint(1, 5)
        used = set()
        for _ in range(num_items):
            p = random.choice(prods)
            if p.id in used:
                continue
            used.add(p.id)
            qty = random.randint(1, 15)
            soi = SalesOrderItem(sales_order_id=so.id, product_id=p.id, quantity=qty, unit_price=p.unit_price)
            db.add(soi)
            subtotal += qty * p.unit_price
        so.subtotal = subtotal
        so.tax_amount = round(subtotal * tax_rate / 100)
        so.total_amount = subtotal + so.tax_amount
        sos.append(so)
    db.flush()

    # Invoices (from INVOICED sales orders)
    inv_statuses = ["PAID", "PAID", "PAID", "PAID", "PAID", "SENT", "SENT", "OVERDUE", "DRAFT", "PAID",
                    "PAID", "PAID", "PAID", "PAID"]
    inv_idx = 0
    for so in sos:
        if so.status == "INVOICED":
            st = inv_statuses[inv_idx % len(inv_statuses)]
            days_ago = random.randint(10, 300)
            created = datetime.utcnow() - timedelta(days=days_ago)
            due = (created + timedelta(days=30)).date()
            inv = Invoice(
                sales_order_id=so.id, invoice_number=f"INV-{inv_idx+1:04d}", status=st,
                subtotal=so.subtotal, tax_amount=so.tax_amount, total_amount=so.total_amount,
                due_date=due, created_at=created,
            )
            db.add(inv)
            db.flush()
            # Copy items
            for soi_row in db.query(SalesOrderItem).filter(SalesOrderItem.sales_order_id == so.id).all():
                prod = db.query(Product).get(soi_row.product_id)
                db.add(InvoiceItem(
                    invoice_id=inv.id, description=prod.name if prod else "Item",
                    quantity=soi_row.quantity, unit_price=soi_row.unit_price,
                    amount=soi_row.quantity * soi_row.unit_price,
                ))
            # Update account balances for PAID invoices
            if st == "PAID":
                accs["1100"].balance += inv.total_amount  # Cash in
                accs["4000"].balance += inv.total_amount  # Revenue
                # Journal entries
                db.add(JournalEntry(entry_date=due, description=f"Invoice {inv.invoice_number} issued",
                                    debit_account_id=accs["1300"].id, credit_account_id=accs["4000"].id,
                                    amount=inv.total_amount, reference_type="invoice", reference_id=inv.id))
                db.add(JournalEntry(entry_date=due + timedelta(days=random.randint(1, 20)),
                                    description=f"Payment for {inv.invoice_number}",
                                    debit_account_id=accs["1100"].id, credit_account_id=accs["1300"].id,
                                    amount=inv.total_amount, reference_type="invoice", reference_id=inv.id))
            elif st in ("SENT", "OVERDUE"):
                accs["1300"].balance += inv.total_amount
                accs["4000"].balance += inv.total_amount
                db.add(JournalEntry(entry_date=due, description=f"Invoice {inv.invoice_number} issued",
                                    debit_account_id=accs["1300"].id, credit_account_id=accs["4000"].id,
                                    amount=inv.total_amount, reference_type="invoice", reference_id=inv.id))
            inv_idx += 1
    db.flush()

    db.commit()
    print("✅ Database seeded with sample data.")


# ─── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
