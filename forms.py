from flask_wtf import Form
from wtforms import TextField, SubmitField

class SearchForm(Form):
    query = TextField('Enter your text here')
    submit = SubmitField('Search')