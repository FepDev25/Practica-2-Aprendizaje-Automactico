from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL

url = URL.create(
    drivername= "postgresql",
    username= "postgres",
    password="root",
    host="localhost",
    database= "db",
    port=5432
)
# url = URL.create(
#     drivername= "postgresql",
#     username= "admin",
#     password="admin1234",
#     host="localhost",
#     database= "db",
#     port = 4040
# )

engine = create_engine(url=url, echo=True)

Session = sessionmaker(bind=engine)

session = Session()
