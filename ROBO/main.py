import nltk
import random
import string  # to process standard python strings

f = open('chatbot.txt', 'r', errors='ignore')

raw = f.read()

raw = raw.lower()  # converts to lowercase

nltk.download('punkt')  # first-time use only
nltk.download('wordnet')  # first-time use only

sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

sent_tokens[:2]
['a chatbot (also known as a talkbot, chatterbot, bot, im bot, interactive agent, or artificial conversational entity) is a computer program or an artificial intelligence which conducts a conversation via auditory or textual methods.',
 'such programs are often designed to convincingly simulate how a human would behave as a conversational partner, thereby passing the turing test.']

word_tokens[:2]
['a', 'chatbot', '(', 'also', 'known']

lemmer = nltk.stem.WordNetLemmatizer()
# WordNet is a semantically-oriented dictionary of English included in NLTK.

def LemTokens(tokens): # takes a list of tokens and returns normalized list of tokens
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return \
        LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

responses = ("hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me",["hello", "hi", "greetings", "sup", "what's up", "hey"])

def greeting(sentence): # if user's input is a greeting, return a greeting response
    for word in sentence.split():
        if word.lower() in responses:
            return random.choice(responses)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

flag = True
print("ROBO: My name is Robo. I am your assistant. If you want to exit, type Bye!")

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("ROBO: You are welcome..")
        else:
            if(greeting(user_response) != None):
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! take care..")
