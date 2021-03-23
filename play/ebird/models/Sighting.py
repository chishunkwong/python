from sqlalchemy import Column, Integer, String, Sequence, DateTime, Numeric
from sqlalchemy.orm import declarative_base

# declarative base class
Base = declarative_base()
sighting_id_seq = Sequence('sighting_id_seq')


class Sighting(Base):
    __tablename__ = 'sighting'
    id = Column('sighting_id', Integer, sighting_id_seq,
                server_default=sighting_id_seq.next_value(), primary_key=True)
    species_code = Column(String(10))
    genus = Column(String(40))
    common_name = Column(String(100))
    country = Column(String(10))
    region = Column(String(10))
    lat = Column(Numeric())
    lng = Column(Numeric())
    observation_datetime = Column(DateTime())
    how_many = Column(Integer)
    location_id = Column(String(40))
    submitter_id = Column(String(40))

    def __init__(self, species_code, genus, common_name, country, region, lat, lng,
                 observation_datetime, how_many, location_id, submitter_id):
        self.species_code = species_code
        self.genus = genus
        self.common_name = common_name
        self.country = country
        self.region = region
        self.lat = lat
        self.lng = lng
        self.observation_datetime = observation_datetime
        self.how_many = how_many
        self.location_id = location_id
        self.submitter_id = submitter_id


def create_table(engine):
    Base.metadata.create_all(engine)
