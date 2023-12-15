from flask import *
import pickle
import numpy as np

popular_df = pickle.load(open('popular.pickle','rb'))
avg = list(popular_df['Avg-Rating'].values)
avg_rating = []
for i in avg:
    avg_rating.append(round(i,2))

pt = pickle.load(open('pt.pickle','rb'))
popular_df_0 = pickle.load(open('popular_df_0.pickle','rb'))
cosine_similarities = pickle.load(open('cosine_similarities.pickle','rb'))
books = pickle.load(open('books.pickle','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/popularbooks')
def popularbooks():
    return render_template('topbooks.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num-ratings'].values),
                           rating=avg_rating
                           )

@app.route('/recommender')
def index():
    return render_template('index.html')

@app.route('/recommended',methods=['POST'])
def recommended():
    if request.method == 'POST':
        user_input = request.form['user_input']
        try:
            index = np.where(pt.index==user_input)[0][0]
        except:
            return render_template('invalid.html')
        similar_items = sorted(list(enumerate(cosine_similarities[index])),key=lambda x:x[1],reverse=True)[1:13]
        
        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            temp_df_1 = popular_df_0[popular_df_0['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
            item.extend(list(temp_df_1.drop_duplicates('Book-Title')['num-ratings'].values))
            item.extend(list(temp_df_1.drop_duplicates('Book-Title')['Avg-Rating'].values))
            
            data.append(item)
        return render_template('recommended.html',user_input=user_input,data=data)
                
@app.errorhandler(404)
def invalid_route(e):
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)