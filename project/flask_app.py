from time import time
from flask import Flask, render_template, request
from corpus import CVCorpus
from gensim.models import Word2Vec, Doc2Vec


app = Flask(__name__)


# LOAD DATA
with open('DATA_RAW', 'r', encoding='utf-8') as f:
    DATA_RAW = f.read().split('::SEP::')
    
with open('DATA_PARSED', 'r', encoding='utf-8') as f:
    DATA_PARSED = f.read().split('::SEP::') 
    
# SET PATHS TO VSM MODELS
w2v_model = Word2Vec.load(r'w2v_model\araneum_none_fasttextskipgram_300_5_2018.model')
d2v_model = Doc2Vec.load(r'd2v_model\d2v_model.model')

# INSTANTIATE <CVCorpus> object  
CORPUS = CVCorpus(w2v_model=w2v_model, d2v_model=d2v_model)

# BUILD
CORPUS.build(DATA_PARSED, DATA_RAW)
CORPUS.build_inverted_index()


def parse_request(request):
    args = dict()
    args['query'] = request.args.get('query')
    args['algo'] = request.args.get('algo')
    args['top_n'] = int(request.args.get('top_n'))
    args['split_by'] = None if request.args.get('split_by') == 'None' else request.args.get('split_by') 
    args['lemmatize'] = True if 'lemmatize' in request.args else False
    return args
 
    
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET'])
def search():
    start = time()
    results = CORPUS.search(**parse_request(request))
    runtime = '%.2f' % (time() - start)
    return render_template('search.html',
                           query=request.args.get('query'),
                           runtime=runtime,
                           results=results
                          )


@app.route('/view')
def view():
    content = CORPUS.D[int(request.args['cv_id'])].full_content
    return render_template('view.html', content=content)


if __name__ == '__main__':
    app.run()