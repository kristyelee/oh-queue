"""add appointment limits

Revision ID: 3ac7f97f07e4
Revises: 7a9d9550b30f
Create Date: 2020-03-11 04:59:46.480627

"""

# revision identifiers, used by Alembic.
from sqlalchemy import orm

revision = '3ac7f97f07e4'
down_revision = '7a9d9550b30f'

from alembic import op
import sqlalchemy as sa
import oh_queue.models
from oh_queue.models import *


def upgrade():
    # Get alembic DB bind
    connection = op.get_bind()
    session = orm.Session(bind=connection)

    for course in session.query(ConfigEntry.course).distinct():
        session.add(ConfigEntry(key='daily_appointment_limit', value='2', public=True, course=course[0]))
        session.add(ConfigEntry(key='weekly_appointment_limit', value='5', public=True, course=course[0]))

    session.commit()


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    pass
    # ### end Alembic commands ###
