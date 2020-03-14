import numpy as np
import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
import multiprocessing as mp
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin, BaseEstimator
from normalise import normalise
import pandas as pd

stopWords = ["nt","'ll", "'ve", '1-1', 'a', "a's", 'able', 'about', 'above', 'abroad', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'adopted', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'again', 'against', 'ago', 'ah', 'ahead', "ain't", 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'alongside', 'already', 'also', 'although', 'always', 'am', 'amid', 'amidst', 'among', 'amongst', 'amoungst', 'amount', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apart', 'apparently', 'appear', 'appreciate', 'appropriate', 'approximately', 'are', 'area', 'areas', 'aren', "aren't", 'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asked', 'asking', 'asks', 'associated', 'at', 'auth', 'available', 'away', 'awfully', 'b', 'back', 'backed', 'backing', 'backs', 'backward', 'backwards', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'began', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'beings', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'big', 'bill', 'biol', 'both', 'bottom', 'brief', 'briefly', 'but', 'by', 'c', "c'mon", "c's", 'ca', 'call', 'called', 'came', 'can', "can't", 'cannot', 'cant', 'caption', 'case', 'cases', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clear', 'clearly', 'co', 'co.', 'com', 'come', 'comes', 'computer', 'con', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains', 'corresponding', 'could', "couldn't", 'couldnt', 'course', 'cry', 'currently', 'd', 'dare', "daren't", 'date', 'de', 'dear', 'definitely', 'describe', 'described', 'despite', 'detail', 'did', "didn't", 'differ', 'different', 'differently', 'directly', 'do', 'does', "doesn't", 'doing', "don't", 'done', 'down', 'downed', 'downing', 'downs', 'downwards', 'due', 'during', 'e', 'each', 'early', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'end', 'ended', 'ending', 'ends', 'enough', 'entirely', 'especially', 'et', 'et-al', 'etc', 'even', 'evenly', 'ever', 'evermore', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except', 'f', 'face', 'faces', 'fact', 'facts', 'fairly', 'far', 'farther', 'felt', 'few', 'fewer', 'ff', 'fifteen', 'fifth', 'fify', 'fill', 'find', 'finds', 'fire', 'first', 'five', 'fix', 'followed', 'following', 'follows', 'for', 'forever', 'former', 'formerly', 'forth', 'forty', 'forward', 'found', 'four', 'from', 'front', 'full', 'fully', 'further', 'furthered', 'furthering', 'furthermore', 'furthers', 'g', 'gave', 'general', 'generally', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'going', 'gone', 'good', 'goods', 'got', 'gotten', 'great', 'greater', 'greatest', 'greetings', 'group', 'grouped', 'grouping', 'groups', 'h', 'had', "hadn't", 'half', 'happens', 'hardly', 'has', "hasn't", 'hasnt', 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'hed', 'held', 'hello', 'help', 'hence', 'her', 'here', "here's", 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herse', 'herself', 'hes', 'hi', 'hid', 'high', 'higher', 'highest', 'him', 'himse', 'himself', 'his', 'hither', 'home', 'hopefully', 'how', 'howbeit', 'however', 'hundred', 'i', "i'd", "i'll", "i'm", "i've", 'id', 'ie', 'if', 'ignored', 'im', 'immediate', 'immediately', 'importance', 'important', 'in', 'inasmuch', 'inc', 'inc.', 'include', 'included', 'including', 'indeed', 'index', 'indicate', 'indicated', 'indicates', 'information', 'inner', 'inside', 'insofar', 'instead', 'interest', 'interested', 'interesting', 'interests', 'into', 'invention', 'inward', 'is', "isn't", 'it', "it'd", "it'll", "it's", 'itd', 'its', 'itse', 'itself', 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'keys', 'kg', 'kind', 'km', 'knew', 'know', 'known', 'knows', 'l', 'large', 'largely', 'last', 'late', 'lately', 'later', 'latest', 'latter', 'latterly', 'least', 'led', 'less', 'lest', 'let', "let's", 'lets', 'like', 'liked', 'likely', 'likewise', 'line', 'links', 'little', 'long', 'longer', 'longest', 'look', 'looking', 'looks', 'low', 'lower', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'making', 'man', 'many', 'may', 'maybe', "mayn't", 'me', 'mean', 'means', 'meantime', 'meanwhile', 'member', 'members', 'men', 'merely', 'mg', 'might', "mightn't", 'mill', 'million', 'mine', 'minus', 'miss', 'ml', 'more', 'moreover', 'most', 'mostly', 'move', 'moved', 'mr', 'mrs', 'much', 'mug', 'must', "mustn't", 'my', 'myse', 'myself', 'n', 'na', 'name', 'namely', 'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needed', 'needing', "needn't", 'needs', 'neither', 'never', 'neverf', 'neverless', 'nevertheless', 'new', 'newer', 'newest', 'next', 'nine', 'ninety', 'no', 'no-one', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'notwithstanding', 'novel', 'now', 'nowhere', 'number', 'numbers', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'older', 'oldest', 'omitted', 'on', 'once', 'one', "one's", 'ones', 'only', 'onto', 'open', 'opened', 'opening', 'opens', 'opposite', 'or', 'ord', 'order', 'ordered', 'ordering', 'orders', 'other', 'others', 'otherwise', 'ought', "oughtn't", 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'page', 'pages', 'part', 'parted', 'particular', 'particularly', 'parting', 'parts', 'past', 'per', 'perhaps', 'place', 'placed', 'places', 'please', 'plus', 'point', 'pointed', 'pointing', 'points', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'presented', 'presenting', 'presents', 'presumably', 'previously', 'primarily', 'probably', 'problem', 'problems', 'promptly', 'proud', 'provided', 'provides', 'put', 'puts', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'reasonably', 'received', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'respectively', 'resulted', 'resulting', 'results', 'right', 'room', 'rooms', 'round', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sec', 'second', 'secondly', 'seconds', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'sees', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'seven', 'several', 'shall', "shan't", 'she', "she'd", "she'll", "she's", 'shed', 'shes', 'should', "shouldn't", 'show', 'showed', 'showing', 'shown', 'showns', 'shows', 'side', 'sides', 'significant', 'significantly', 'similar', 'similarly', 'since', 'sincere', 'six', 'sixty', 'slightly', 'small', 'smaller', 'smallest', 'so', 'some', 'somebody', 'someday', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'state', 'states', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure', 'system', 't', "t's", 'take', 'taken', 'taking', 'tell', 'ten', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", "that's", "that've", 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', "there'd", "there'll", "there're", "there's", "there've", 'thereafter', 'thereby', 'thered', 'therefore', 'therein', 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'theyd', 'theyre', 'thick', 'thin', 'thing', 'things', 'think', 'thinks', 'third', 'thirty', 'this', 'thorough', 'thoroughly', 'those', 'thou', 'though', 'thoughh', 'thought', 'thoughts', 'thousand', 'three', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'till', 'time', 'tip', 'tis', 'to', 'today', 'together', 'too', 'took', 'top', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'turn', 'turned', 'turning', 'turns', 'twas', 'twelve', 'twenty', 'twice', 'two', 'u', 'un', 'under', 'underneath', 'undoing', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'upwards', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'uucp', 'v', 'value', 'various', 'versus', 'very', 'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wanted', 'wanting', 'wants', 'was', "wasn't", 'way', 'ways', 'we', "we'd", "we'll", "we're", "we've", 'wed', 'welcome', 'well', 'wells', 'went', 'were', "weren't", 'what', "what'll", "what's", "what've", 'whatever', 'whats', 'when', 'whence', 'whenever', 'where', "where's", 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'whichever', 'while', 'whilst', 'whim', 'whither', 'who', "who'd", "who'll", "who's", 'whod', 'whoever', 'whole', 'whom', 'whomever', 'whos', 'whose', 'why', 'widely', 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'wonder', 'words', 'work', 'worked', 'working', 'works', 'world', 'would', "wouldn't", 'written', 'www', 'x', 'y', 'year', 'years', 'yes', 'yet', 'you', "you'd", "you'll", "you're", "you've", 'youd', 'young', 'younger', 'youngest', 'your', 'yourabout', 'youre', 'yours', 'yourself', 'yourselves', 'z', 'zero']


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 variety="BrE",
                 user_abbrevs={},
                 n_jobs=1):
        self.variety = variety
        self.user_abbrevs = user_abbrevs
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self

    def transform(self, X, *_):
        X_copy = X.copy()

        partitions = 1
        cores = mp.cpu_count()
        if self.n_jobs <= -1:
            partitions = cores
        elif self.n_jobs <= 0:
            return X_copy.apply(self._preprocess_text)
        else:
            partitions = min(self.n_jobs, cores)

        data_split = np.array_split(X_copy, partitions)
        pool = mp.Pool(cores)
        data = pd.concat(pool.map(self._preprocess_part, data_split))
        len(data)
        pool.close()
        pool.join()

        return data
    
    def strip_html(self,text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_between_square_brackets(self,text):
        return re.sub('\[[^]]*\]', '', text)

    def denoise_text(self,text):
        text = self.strip_html(text)
        text = self.remove_between_square_brackets(text)
        return text
    
    def remove_non_ascii(self,words):
    
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self,words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self,words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self,words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self,words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english') and word not in stopWords:
                new_words.append(word)
        return new_words

    def stem_words(self,words):
        """Stem words in list of tokenized words"""
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self,words):
        """Lemmatize verbs in list of tokenized words"""
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def normalize2(self,words):
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
#         words = self.replace_numbers(words)
        words = self.remove_stopwords(words)
        return words

    def stem_and_lemmatize(self,words):
        #stems = stem_words(words)
        lemmas = self.lemmatize_verbs(words)
        return lemmas

    def _preprocess_part(self, part):
        return part.apply(self._preprocess_text)

    def _preprocess_text(self, text):
        sample = text
        words = nltk.word_tokenize(sample)
        words = self.normalize2(words)
        words = self.stem_and_lemmatize(words)
        return ' '.join(words)

    def _normalize(self, text):
        # some issues in normalise package
        try:
            return ' '.join(normalise(text, variety=self.variety, user_abbrevs=self.user_abbrevs, verbose=False))
        except:
            return text

    def _remove_punct(self, doc):
        return [t for t in doc if t.text not in string.punctuation]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop and t.doc not in stopWords and t.doc not in ["http","multimedia","html"]]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])


