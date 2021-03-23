"""Script used to preprocess New Yorker captions"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pandas as pd
import re
import spacy
import string
from nltk.corpus import stopwords
from spacy.tokens import Doc

# change this to where https://github.com/nextml/caption-contest-data is cloned to
DATA_DIR = "/home/ubuntu/efs/image_project/caption_contest_data"
NY_SOURCE_DATA_DIR = os.path.join(DATA_DIR, "contests", "info")

# change this to anywhere
OUTPUT_DATA_DIR = "/home/ubuntu/efs/image_project/new_yorker"

# from: https://stackoverflow.com/questions/201323/how-to-validate-an-email-address-using-a-regular-expression
RE_EMAIL = re.compile(r"^\S+@\S+\.\S+$")
RE_HASHTAG = re.compile(r"#\S+")
RE_PARENS = re.compile(r"\(.+\)")


# spacy configuration
class WhitespaceTokenizer:
    """Whitespace Tokenizer for spacy"""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = [w for w in text.split(" ") if len(w) > 0]
        return Doc(self.vocab, words=words)


nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat", "lemmatizer"])
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
stops = set(stopwords.words("english"))
custom_stops = [
    "im",
    "one",
    "like",
    "dont",
    "us",
    "hes",
    "shes" "thats",
    "didnt",
    "get",
    "got",
    "youre",
    "think",
    "said",
    "new",
    "know",
    "back",
    "well",
    "sorry",
    "see",
    "feel",
    "look",
    "come",
    "going",
    "theyre",
    "were",
    "told",
    "yes",
    "i'm",
    "he's",
    "she's",
    "that's",
    "didn't",
    "you're",
    "they're",
    "we're",
]
stops.update(custom_stops)

# main method below
def get_file_id_to_captions(caption_starts, max_length):
    """Get a map from file_id to a list of captions

    Captions are filtered by caption_starts and max_length

    Parameters
    ----------
    caption_starts: set
        set of strings that a caption must start with
    max_length: int
        max number of tokens allowed in a caption

    """
    file_id_to_description = get_file_id_to_description()
    file_id_to_nouns = get_file_id_to_nouns()

    first_n = 10  # categories must appear in the first n words
    file_id_to_captions = dict()
    for file_id in os.listdir(NY_SOURCE_DATA_DIR):
        if not str(file_id).isnumeric() or int(file_id) not in file_id_to_description:
            continue
        captions = get_captions(file_id)
        if len(captions) == 0:
            continue
        pp_categories = file_id_to_nouns[int(file_id)]
        pp_captions = [preprocess_caption(c) for c in captions]
        pp_captions = [c for c in pp_captions if c is not None]
        pp_captions_start = [
            c for c in pp_captions if len(c) > 0 and c.split()[0] in caption_starts
        ]
        filtered_captions = []
        for caption in pp_captions_start:
            caption_tokens = caption.split()
            if len(set(caption_tokens[0:first_n]).intersection(pp_categories)) >= min(
                2, len(pp_categories)
            ):
                filtered_captions.append(caption)
        if len(filtered_captions) == 0:
            for caption in pp_captions_start:
                if len(
                    set(caption.split()[0:first_n]).intersection(pp_categories)
                ) >= min(1, len(pp_categories)):
                    filtered_captions.append(caption)
        filtered_captions = [
            c for c in filtered_captions if len(c.split()) <= max_length
        ]

        if len(filtered_captions) == 0:
            print("No suitable captions found for", file_id)
        file_id_to_captions[int(file_id)] = filtered_captions
    return file_id_to_captions


def extract_nouns(string):
    """Extract all the nouns from the given string"""
    nouns = []
    doc = nlp(string)
    for token in doc:
        if token.pos_ == "PROPN" or token.pos_ == "NOUN":
            nouns.append(token.text)
    return nouns


def has_numbers(string):
    """Returns true if there are digits in the string"""
    return any(char.isdigit() for char in string)


def preprocess_caption(caption):
    """Preprocess the provided caption"""
    # some captions are just an email
    if RE_EMAIL.match(caption):
        return None

    # remove attribution emails (they appear after semicolons)
    caption = caption.split(";")[0]

    # remove hashtags and parentheses
    caption = RE_HASHTAG.sub(" ", caption)
    caption = RE_PARENS.sub(" ", caption)

    # some captions are bad; these contain numbers
    if has_numbers(caption):
        return None

    # remove punctuation
    caption = re.sub(r"[’']+", "", caption)
    caption = re.sub(r"[’'\-\"\.,\—\?_…]+", " ", caption)
    caption = (
        str(caption)
        .lower()
        .translate(str.maketrans("", "", string.punctuation))
        .strip()
    )

    # conflate spaces
    return re.sub(r"\s+", " ", caption)


def file_id_to_fname(file_id):
    """Returns the image filename for the given file_id"""
    return os.path.join(NY_SOURCE_DATA_DIR, f"{file_id}/{file_id}.jpg")


def display_image(file_id):
    """Display the image with the provided file_id"""
    filepath = file_id_to_fname(file_id)
    print(filepath)
    image = mpimg.imread(filepath)
    plt.imshow(image)
    plt.show()


def get_captions(file_id):
    """Get captions for the provided file_id

    Returns emtpy array if no caption file found

    """
    captions = []

    # the captions can be stored in a couple of places. Try them all.
    fname_txt = os.path.join(
        NY_SOURCE_DATA_DIR, str(file_id), f"{file_id}_captions.txt"
    )
    fname_txt2 = os.path.join(
        NY_SOURCE_DATA_DIR, str(file_id), f"{file_id}_captions_output.txt"
    )
    fname_csv = os.path.join(
        NY_SOURCE_DATA_DIR, str(file_id), f"{file_id}_captions.csv"
    )
    fname_csv2 = os.path.join(
        NY_SOURCE_DATA_DIR, str(file_id), f"{file_id}_captions_output.csv"
    )
    if os.path.isfile(fname_txt):
        fname = fname_txt
    elif os.path.isfile(fname_txt2):
        fname = fname_txt2
    elif os.path.isfile(fname_csv):
        fname = fname_csv
    elif os.path.isfile(fname_csv2):
        fname = fname_csv2
    else:
        print("Path not found for", file_id)
        return []

    # read captions
    with open(fname, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            if len(line) > 0:
                captions.append(line.strip())
    return captions


def get_file_id_to_description():
    """Return a map from file_id to a description of the file"""
    df_descriptions = pd.read_csv(
        os.path.join(DATA_DIR, "contests/metadata/descriptions.txt")
    )
    file_id_to_description = df_descriptions.set_index("contest").to_dict()[
        "description"
    ]
    return file_id_to_description


def get_file_id_to_nouns():
    """Return a map from file_id to the objects in the image"""
    file_id_to_description = get_file_id_to_description()
    file_id_to_nouns = dict()
    for file_id, desc in file_id_to_description.items():
        nouns = extract_nouns(desc)
        pp_nouns = [preprocess_caption(noun) for noun in nouns]
        filtered_nouns = [noun for noun in pp_nouns if noun not in stops]
        file_id_to_nouns[file_id] = set(filtered_nouns)
    return file_id_to_nouns
