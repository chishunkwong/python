from .models.Sighting import Sighting
from .models import Sighting as sight
from . import ebird_api as eb
import datetime as dt
import json
import sqlalchemy
from sqlalchemy.orm import sessionmaker

db = 'DB'


def connect():
    '''Returns an engine, a metadata object and a Session'''
    with open('db_config.json') as f:
        data = json.load(f)
        db = data['db']
        host = data['host']
        port = data['port']
        username = data['username']
        password = data['password']
    # We connect with the help of the PostgreSQL URL
    # postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(username, password, host, port, db)

    # The return value of create_engine() is our connection object
    engine = sqlalchemy.create_engine(url, client_encoding='utf8')
    # We then bind the connection to MetaData()
    meta = sqlalchemy.MetaData(bind=engine)
    Session = sessionmaker(bind=engine)

    return engine, meta, Session


def create_tables():
    engine, meta, Session = connect()
    sight.create_table(engine)


def insert(country, region):
    engine, _, Session = connect()
    session = Session()
    res_json = eb.get_region_recent_sightings(country, region)
    for s in res_json:
        sciName = s['sciName']
        genus = sciName.split(' ', 1)[0]
        obsDt = s['obsDt']
        date_format = '%Y-%m-%d' if len(obsDt) == 10 else '%Y-%m-%d %H:%M'
        observation_datetime = dt.datetime.strptime(obsDt, date_format)
        sig = Sighting(s['speciesCode'], genus, s['comName'],
                       country, region, s['lat'], s['lng'],
                       observation_datetime, s['howMany'], s['locId'], s['subId'])
        session.add(sig)
    session.commit()


