from flask import Flask, render_template, session, redirect, request, url_for
from werkzeug.utils import secure_filename
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import model_from_json
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# from keras.models import model_from_yaml
# import pandas as pd
# import numpy
from forms import SearchForm
import os
import csv
# from flask import Flask
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
UPLOAD_FOLDER = app.root_path + '/static/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

di = {'Acne' : [ {"name": "Afol","price" : "13"}, {"name": "Aldactone","price" : "25"}, 
 {"name": "AIMIL","price" : "25"}, {"name": "Pernex","price" : "47"}, 
 {"name": "Perobar","price" : "107"}, {"name": "Bengel AC","price" : "111"}],
  'Psoriasis' : [{"name": "Vitfol","price" : "10"}, {"name": "Naprosyn","price" : "22"}, 
  {"name": "Naproz","price" : "34"},{"name": "Pernex","price" : "47"},
 {"name": "Arthopan","price" : "52"}, {"name": "Fecotin","price" : "143"}, ]}

@app.route("/")
def home():
	return render_template('home.html')

@app.route("/home")
def index():
	session['version'] = 1
	session['di'] = di
	# with open("Salt_Disease.csv","r") as f:
	# 	rdr = csv.reader(f)
	# 	rdr.next()
	# 	for row in rdr:
	# 			with open("medicine_disease.csv","r") as f2:
	# 				rdr2 = csv.reader(f2)
	# 				rdr2.next()
	# 				for row2 in rdr2:
	# 					if row2[0] in row[0]:
	# 						if row[1] in di.keys():
	# 							di[row[1]].append({"name" : row2[1] , "price" : row2[2]})
	# 						else:
	# 							di[row[1]] = [{"name" : row2[1] , "price" : row2[2]}]

	# # for key in di:
	# # 	print key, di[key]
	# session['di'] = di
	# print di['Acne']
	# print app.config['UPLOAD_FOLDER']
	return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	# print app.config['UPLOAD_FOLDER']
	# print 'abc'

	if request.method == 'POST':
		# print 'abc'
		print request.files
		# if not request.form['file']:
		# 	print 1
		# else:
		# 	print request.form['file']
		# check if the post request has the file part
		# if file: 
   #		  # load json and create model
			# json_file = open('model.json', 'r')
			# loaded_model_json = json_file.read()
			# json_file.close()
			# loaded_model = model_from_json(loaded_model_json)
			# # load weights into new model
			# loaded_model.load_weights("model.h5")
			# print("Loaded model from disk")
		# print file
		# print request.forms.post('file')
		# print request.forms['file']
		file = request.files['file']
		filename = 'tmp.jpg'
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		# image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		# im = cv2.imread(image_path,1)
		# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		# img = cv2.resize(im, (224, 224))
		# imag = np.asarray(imag)
		# imag = np.reshape(imag,(1,224,224,3))

		# ans2 = model.predict(imag)
		# ans =  np.argmax(ans2)

		ans = 0
		# print 
		# print filename
		return redirect(url_for('uploaded_file',filename=str(filename), ans = str(ans)))


@app.route("/search", methods = ['GET','POST'])
def search():
	form = SearchForm()
	if form.validate_on_submit():
		# print 'aaa'
		x = {}
		y = {}
		query = form.query.data
		# print query	
		for key in di:
			for row in di[key]:
				# print row
				if query == row['name']:
					y[query] = row['price']
					# print row['name']
					for row2 in di[key]:
						if query != row2['name']:
							if int(row2['price']) < int(y[query]):
								x[row2['name']] = row2['price']
			# print x
		return render_template("search.html", form = form, x = x, y=y)
	return render_template("search_main.html", form = form)


@app.route("/uploaded_file/<filename>/<ans>", methods = ['GET'])
def uploaded_file(filename, ans):

	# player_name = request.json['name']
	# session['player'] = player_name
	# print i
	# return render_template(name.html)
	# return "hello"

	# return "hello"
	if ans == '0':
		ans = 'Acne'
	else:
		ans = 'Sporasis'

	di = session.get('di')
	
	version = session.get('version')
	session['version'] += 1
	# with open('static/css/main.css','r') as f:
	# 	with open('static/css/main?version=' + str(version) + '.css','w') as w:
	# 		for row in f:
	# 			w.write(row)


	return render_template('final.html', version=str(version), filename=filename, ans=ans, di=di)

if __name__ == "__main__":
	app.secret_key = 'super secret key'
	app.config['SESSION_TYPE'] = 'filesystem'

	# sess.init_app(app)
	app.run(host='0.0.0.0',port=5004, threaded=True)