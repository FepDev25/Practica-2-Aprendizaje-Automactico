from sqlalchemy import Column, Integer, String, Float, Boolean, Date
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Registro(Base):
    __tablename__ = "registros"

    id = Column(Integer, primary_key=True, autoincrement=True)

    created_at = Column(Date)
    product_id = Column(Integer)
    product_name = Column(String(255))
    product_sku = Column(String(100))
    supplier_id = Column(Integer)
    supplier_name = Column(String(255))
    prioridad_proveedor = Column(Integer)

    quantity_on_hand = Column(Integer)
    quantity_reserved = Column(Integer)
    quantity_available = Column(Integer)

    minimum_stock_level = Column(Integer)
    reorder_point = Column(Integer)
    optimal_stock_level = Column(Integer)
    reorder_quantity = Column(Integer)

    average_daily_usage = Column(Float)

    last_order_date = Column(Date)
    last_stock_count_date = Column(Date)

    unit_cost = Column(Float)
    total_value = Column(Float)

    expiration_date = Column(Date)

    batch_number = Column(String(100))
    warehouse_location = Column(String(200))
    shelf_location = Column(String(200))
    region_almacen = Column(String(200))

    stock_status = Column(String(50))
    is_active = Column(Boolean)

    last_updated_at = Column(Date)

    created_by_id = Column(Integer)
    record_sequence_number = Column(Integer)

    categoria_producto = Column(String(200))
    subcategoria_producto = Column(String(200))

    anio = Column(Integer)
    mes = Column(Integer)

    vacaciones_o_no = Column(Boolean)
    es_feriado = Column(Boolean)
    temporada_alta = Column(Boolean)
