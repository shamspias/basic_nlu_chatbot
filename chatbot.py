from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

### Download for 1st time ###
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

# Define a list of stopwords to remove
# stop_words = set(stopwords.words('english'))


# Define the training data

greeting_data = [("Hey", "greeting"),
                ("Hello", "greeting"),
                ("How are you?", "greeting"),
                ("What's up?", "greeting"),
                ("How's it going?", "greeting"),
                ("Good morning", "greeting"),
                ("Good afternoon", "greeting"),
                ("Good evening", "greeting"),
                ("Hi", "greeting"),
                ("Hey there", "greeting"),
                ("What's new?", "greeting"),
                ("How's everything?", "greeting"),
                ("How's your day?", "greeting"),
                ("How's your week?", "greeting"),
                ("How's life?", "greeting"),
                ("Nice to meet you", "greeting"),
                ("Nice to see you", "greeting"),
                ("Long time no see", "greeting"),
                ("It's been a while", "greeting"),
                ("What's been going on?", "greeting"),
                ("How have you been?", "greeting"),
                ("What's been happening?", "greeting"),
                ("How's your day going?", "greeting"),
                ("How's your week going?", "greeting"),
                ("How's your month going?", "greeting"),
                ("How's your year going?", "greeting"),
                ("How's your life going?", "greeting"),
                ("What's new with you?", "greeting"),
                ("What's been up with you?", "greeting"),
                ("How's everything with you?", "greeting"),
                ("How have things been?", "greeting"),
                ("What's been happening with you?", "greeting"),
                ("How's your day been?", "greeting"),
                ("How's your week been?", "greeting"),
                ("How's your month been?", "greeting"),
                ("How's your year been?", "greeting"),
                ("How's your life been?", "greeting"),
                ("What's new in your life?", "greeting"),
                ("What's been happening in your life?", "greeting"),
                ("How's everything in your life?", "greeting"),
                ("How have things been in your life?", "greeting"),
                ("How's your day been going?", "greeting"),
                ("How's your week been going?", "greeting"),
                ("How's your month been going?", "greeting"),
                ("How's your year been going?", "greeting"),
                ("How's your life been going?", "greeting"),
                ("What's new with you?", "greeting"),
                ("What's been up with you?", "greeting"),
                ("How's everything with you?", "greeting"),
                ("How have you been?", "greeting"),
                ("What's been happening?", "greeting"),
                ("How's your day going?", "greeting"),
                ("How's your week going?", "greeting"),
                ("How's your month going?", "greeting"),
                ("How's your year going?", "greeting"),
                ("How's your life going?", "greeting"),
                ("What's new in your life?", "greeting"),
                ("What's been happening in your life?", "greeting"),
                ("How's everything in your life?", "greeting"),
                ("How have things been in your life?", "greeting"),
                ("How's your day been going?", "greeting"),]    

account_data = [("I need to reset my password", "account"),
                ("I can't log into my account", "account"),
                ("I want to update my account information", "account")]

shipping_data = [("When will my order be shipped?", "shipping"),
                 ("How can I track my package?", "shipping"),
                 ("Can you expedite my shipping?", "shipping")]

refund_data = [("I want a refund for my order", "refund"),
               ("I am not satisfied with my purchase and would like my money back", "refund"),
               ("I would like to return my item for a full refund", "refund")]

complaints_data = [("Your service is terrible", "complaint"),
                   ("I am extremely dissatisfied with your product", "complaint"),
                   ("I am not happy with my purchase", "complaint"),
                   ("I received a damaged product", "complaint")]

basic_data = [("I am having trouble with my account", "problem"),
                 ("I need help with my order", "problem"),
                 ("I want to cancel my subscription", "problem"),
                 ("Thank you for your assistance", "thanks"),
                 ("I am satisfied with your service", "satisfaction"),
                 ("I appreciate your help", "thanks")]

training_data = account_data + shipping_data + refund_data + complaints_data + basic_data + greeting_data

# Tokenize the input text
training_data = [(word_tokenize(text), label) for text, label in training_data]

# Create a frequency distribution of the input text
fdist = FreqDist()
for text, label in training_data:
    for token in text:
        fdist[token] += 1

# Create a dictionary of features
def create_features(text):
    return {token: True for token in text if fdist[token] > 1}

# Train the classifier
classifier = NaiveBayesClassifier.train([(create_features(text), label) for text, label in training_data])

# Define the function to handle customer input
def handle_input(customer_input):
    # Tokenize the input
    input_tokens = word_tokenize(customer_input)
    
    # Create the feature set
    input_features = create_features(input_tokens)
    
    # Classify the input
    classification = classifier.classify(input_features)
    
    # Respond based on the classification
    if classification == "problem":
        return "I'm sorry to hear that you're having a problem. Can you please provide more details so I can assist you?"
    elif classification == "thanks":
        return "You're welcome! Is there anything else I can help you with?"
    elif classification == "greeting":
        return "I am fine how about you?"
    elif classification == "satisfaction":
        return "I'm glad to hear that you're satisfied with our service. Let me know if there's anything else I can do for you."
    else:
        return "I'm not sure I understand. Can you please rephrase that?"

# Test the function
user_input = input("Enter yur question: ")
print(handle_input(user_input))
# Output: "I'm sorry to hear that you're having a problem. Can you please provide more details so I can assist you?"
