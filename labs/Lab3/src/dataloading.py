from torch.utils.data import Dataset
from tqdm import tqdm
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(s):
    s = s.strip()  # strip leading / trailing spaces
    s = s.lower()  # convert to lowercase
    s = contractions.fix(s)  # e.g. don't -> do not, you're -> you are
    s = re.sub("\s+", " ", s)  # strip multiple whitespace
    s = re.sub(r"[^a-z0-9A-Z\s]", " ", s)  # keep only lowercase letters and spaces

    return s

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        
        # 6.1 Implement tf-idf vectorizer of dataset
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit_transform(X)
        
    
        # EX2
        for i in range(len(X)):
            X[i] = clean_text(X[i])
            X[i] = [w for w in X[i].split(" ") if len(w) > 0]

        self.data = X
        self.labels = y
        self.word2idx = word2idx

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
        sentence = self.data[index]
        max_dimension = 22 #chosen from average length
        length = len(sentence)
        example=[]
        tfidf = []
        
        if (len(sentence)>=max_dimension):
            for i in range(max_dimension):
                y = self.vectorizer.vocabulary_.get(sentence[i])
                if (y==None):
                    tfidf.append(0)
                else:
                    tfidf.append(y)
                if sentence[i] in self.word2idx:
                    example.append(self.word2idx[sentence[i]])
                else:
                    example.append(self.word2idx["<unk>"]) #not in dict
            example = example[:(max_dimension)]
            length = max_dimension
        
        if (len(sentence)<max_dimension):
            for i in range(len(sentence)):
                y = self.vectorizer.vocabulary_.get(sentence[i])
                if (y==None):
                    tfidf.append(0)
                else:
                    tfidf.append(y)
                if sentence[i] in self.word2idx:
                    example.append(self.word2idx[sentence[i]])
                else:
                    example.append(self.word2idx["<unk>"]) #not in dict
            for j in range(len(sentence),max_dimension):
                example.append(0) #zero-padding
                tfidf.append(0)

            #example = sentence
        
        label = self.labels[index]
        return example, label, length, tfidf
        #raise NotImplementedError

