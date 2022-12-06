#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:22:36 2022

@author: sergey
"""

import uuid
from flask import Flask, render_template, url_for, request, flash, session, redirect
import numpy as np
from tensorflow.keras import models


import pickle

mod_name1 = 'oneHotEncoder1.pkl'
with open(mod_name1 , 'rb') as f:
    onehotencoder1 = pickle.load(f)

mod_name2 = 'oneHotEncoder2.pkl'
with open(mod_name2 , 'rb') as f:
    onehotencoder2 = pickle.load(f)

mod_name3 = 'scaler.pkl'
with open(mod_name3 , 'rb') as f:
    scaler = pickle.load(f)

model = models.load_model('creditscore.h5')


#print(type(model))





app = Flask(__name__)
app.config['SECRET_KEY'] = 'nsdndjngjnnjdasjkwebju5489djkdgjkdjk347hdk'

# Словарь пунктов меню и соответствующих им ссылок
menu = [{'name': 'Ввод нового события', 'url': 'enter_item'},
        {'name': 'О проекте', 'url': 'about'} ]



# Обработчик главной страницы
@app.route("/")
@app.route("/index")
def index():
    print (url_for('index'))
    return render_template("index.html", title = "Главная страница", menu = menu)

@app.route("/about")
def about():
    return render_template("about.html", title = "о проекте", menu = menu)


# Обработчик формы ввода нового события
# реализует базовые проверки на непустоту полей
@app.route("/enter_item", methods = ['POST', 'GET'])
def enter_item():




    if request.method == 'POST':
        pairs = dict(request.form)


        flags = {}
        for v in pairs.values():
            flags[v] = False

        for k, v in pairs.items():
            if len(v) < 1:
                s = k + ' incorrect'
                flash(s)
            else:
                flags[v] = True

        f = False
        for v in flags.values():
            if v == True:
                f = True


        if f:

            #  подготовка вектора V
            v = []

            for val in pairs.keys():
                if val not in ['Occupation', 'PaymentBehaviour']:
                    v.append(float(pairs[val]))


            print (pairs['Occupation'], pairs['PaymentBehaviour'])

            occ = pairs['Occupation']
            print (type(occ))
            occ = int(occ)
            print (type(occ))

            beh = int (pairs['PaymentBehaviour'])

            occ_enc = onehotencoder1.transform([[occ]]).toarray()
            beh_enc = onehotencoder2.transform([[beh]]).toarray()

            v = np.array(v)
            print (v.shape)
            v = np.concatenate([v,occ_enc[0],beh_enc[0]])
            v = np.expand_dims(v, axis = 0)
 #           v = np.array(list(pairs.values())).reshape(1, -1)
            print (v)
            v = scaler.transform(v)


            pred = model.predict(v)[0]

            cl_pred = pred.argmax()
            answer = ['CreditScore is Good', 'CreditScore is Standard', 'CreditScore is Poor']

            flash(answer[cl_pred])

        else:
            print ('чего-то не хватило')

        return render_template("enter_item.html", title = "новый прогноз", menu = menu)
    else:
        return render_template("enter_item.html", title = "новый прогноз", menu = menu)


if __name__ == '__main__':
    app.run(debug=True)
