DROP TABLE IF EXISTS user_roles CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS roles CASCADE;

CREATE TABLE users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    enabled BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE roles (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    description VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_roles (
    user_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
);

INSERT INTO roles (name, description) VALUES
    ('ROLE_ADMIN', 'Administrador del sistema con todos los permisos'),
    ('ROLE_USER', 'Usuario estándar con permisos básicos'),
    ('ROLE_MODERATOR', 'Moderador con permisos intermedios');

INSERT INTO users (username, password, email, enabled) VALUES
    ('admin', '$2a$10$xn3LI/AjqicFYZFruSwve.681477XaVNaUQbr1gioaWPn4t1KsnmG', 'admin@example.com', true),
    ('usuario1', '$2a$10$xn3LI/AjqicFYZFruSwve.681477XaVNaUQbr1gioaWPn4t1KsnmG', 'usuario1@example.com', true),
    ('usuario2', '$2a$10$xn3LI/AjqicFYZFruSwve.681477XaVNaUQbr1gioaWPn4t1KsnmG', 'usuario2@example.com', true),
    ('moderador', '$2a$10$xn3LI/AjqicFYZFruSwve.681477XaVNaUQbr1gioaWPn4t1KsnmG', 'moderador@example.com', true),
    ('inactivo', '$2a$10$xn3LI/AjqicFYZFruSwve.681477XaVNaUQbr1gioaWPn4t1KsnmG', 'inactivo@example.com', false);

INSERT INTO user_roles (user_id, role_id) VALUES
    (1, 1), -- admin -> ROLE_ADMIN
    (1, 2), -- admin -> ROLE_USER
    (2, 2), -- usuario1 -> ROLE_USER
    (3, 2), -- usuario2 -> ROLE_USER
    (4, 3), -- moderador -> ROLE_MODERATOR
    (4, 2), -- moderador -> ROLE_USER
    (5, 2); -- inactivo -> ROLE_USER

SELECT u.id, u.username, u.email, u.enabled, r.name as role
FROM users u
INNER JOIN user_roles ur ON u.id = ur.user_id
INNER JOIN roles r ON ur.role_id = r.id
ORDER BY u.id;