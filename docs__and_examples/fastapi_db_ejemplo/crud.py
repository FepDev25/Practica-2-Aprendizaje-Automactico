"""
Operaciones CRUD para la base de datos
"""
from sqlalchemy.orm import Session
import bcrypt
import models
import schemas

def hash_password(password: str) -> str:
    """
    Hashear una contraseña usando bcrypt
    BCrypt tiene un límite de 72 bytes, así que truncamos si es necesario
    """
    # Convertir a bytes y truncar a 72 bytes si es necesario
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    # Generar salt y hashear
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Retornar como string
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verificar una contraseña
    Aplicamos el mismo truncamiento que al hashear
    """
    # Convertir a bytes y truncar a 72 bytes si es necesario
    password_bytes = plain_password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    
    # Verificar
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)

# ==================== CRUD DE USUARIOS ====================

def get_user(db: Session, user_id: int):
    """Obtener un usuario por ID"""
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    """Obtener un usuario por username"""
    return db.query(models.User).filter(models.User.username == username).first()

def get_user_by_email(db: Session, email: str):
    """Obtener un usuario por email"""
    return db.query(models.User).filter(models.User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    """Obtener lista de usuarios con paginación"""
    return db.query(models.User).offset(skip).limit(limit).all()

def create_user(db: Session, user: schemas.UserCreate):
    """Crear un nuevo usuario"""
    # Hashear la contraseña
    hashed_password = hash_password(user.password)
    
    # Crear el usuario
    db_user = models.User(
        username=user.username,
        email=user.email,
        password=hashed_password,
        enabled=user.enabled
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user(db: Session, user_id: int, user: schemas.UserUpdate):
    """Actualizar un usuario existente"""
    db_user = get_user(db, user_id)
    
    if db_user is None:
        return None
    
    # Actualizar solo los campos que se enviaron
    update_data = user.model_dump(exclude_unset=True)
    
    # Si se está actualizando la contraseña, hashearla
    if "password" in update_data:
        update_data["password"] = hash_password(update_data["password"])
    
    for key, value in update_data.items():
        setattr(db_user, key, value)
    
    db.commit()
    db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    """Eliminar un usuario"""
    db_user = get_user(db, user_id)
    
    if db_user is None:
        return False
    
    db.delete(db_user)
    db.commit()
    return True

def toggle_user_status(db: Session, user_id: int, enabled: bool):
    """Habilitar o deshabilitar un usuario"""
    db_user = get_user(db, user_id)
    
    if db_user is None:
        return None
    
    db_user.enabled = enabled
    db.commit()
    db.refresh(db_user)
    return db_user

# ==================== CRUD DE ROLES ====================

def get_role(db: Session, role_id: int):
    """Obtener un rol por ID"""
    return db.query(models.Role).filter(models.Role.id == role_id).first()

def get_role_by_name(db: Session, name: str):
    """Obtener un rol por nombre"""
    return db.query(models.Role).filter(models.Role.name == name).first()

def get_roles(db: Session):
    """Obtener todos los roles"""
    return db.query(models.Role).all()

def create_role(db: Session, role: schemas.RoleCreate):
    """Crear un nuevo rol"""
    db_role = models.Role(
        name=role.name,
        description=role.description
    )
    
    db.add(db_role)
    db.commit()
    db.refresh(db_role)
    return db_role

# ==================== CRUD DE USER_ROLES ====================

def add_role_to_user(db: Session, user_id: int, role_id: int):
    """Asignar un rol a un usuario"""
    db_user = get_user(db, user_id)
    db_role = get_role(db, role_id)
    
    if db_user is None or db_role is None:
        return None
    
    # Verificar si el usuario ya tiene el rol
    if db_role not in db_user.roles:
        db_user.roles.append(db_role)
        db.commit()
        db.refresh(db_user)
    
    return db_user

def remove_role_from_user(db: Session, user_id: int, role_id: int):
    """Remover un rol de un usuario"""
    db_user = get_user(db, user_id)
    db_role = get_role(db, role_id)
    
    if db_user is None or db_role is None:
        return None
    
    # Verificar si el usuario tiene el rol
    if db_role in db_user.roles:
        db_user.roles.remove(db_role)
        db.commit()
        db.refresh(db_user)
    
    return db_user
