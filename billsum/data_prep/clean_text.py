"""
A module for storing text-preprocessing functionality.
Resposible for cleaning up the  unnecessary characters/noise from text
"""
import jsonlines
import os
import pandas as pd
import pickle
import re


def replace_semicolon(text, threshold=10):
    '''
    Get rid of semicolons.

    First split text into fragments between the semicolons. If the fragment
    is longer than the threshold, turn the semicolon into a period. O.w treat
    it as a comma.

    Returns new text
    '''
    new_text = ""
    for subset in re.split(';', text):
        subset = subset.strip()  # Clear off spaces
        # Check word count
        if len(subset.split()) > threshold:
            # Turn first char into uppercase
            new_text += ". " + subset[0].upper() + subset[1:]
        else:
            # Just append with a comma
            new_text += ", " + subset

    return new_text


USC_re = re.compile('[Uu]\.*[Ss]\.*[Cc]\.]+')
PAREN_re = re.compile('\([^(]+\ [^\(]+\)')
HTML_RE = re.compile('<.*?>')
BAD_PUNCT_RE = re.compile(r'([%s])' % re.escape(
    '"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
BULLET_RE = re.compile('\n[\ \t]*`*\([a-zA-Z0-9]*\)')
DASH_RE = re.compile('--+')
WHITESPACE_RE = re.compile('\s+')
EMPTY_SENT_RE = re.compile('[,\.]\ *[\.,]')
FIX_START_RE = re.compile('^[^A-Za-z]*')
FIX_PERIOD = re.compile('\.([A-Za-z])')
SECTION_HEADER_RE = re.compile(
    'SECTION [0-9]{1,2}\.|\nSEC\.* [0-9]{1,2}\.|Sec\.* [0-9]{1,2}\.')


def clean_text(text):
    """
    Borrowed from the FNDS text processing with additional logic added in.
    Note: we do not take care of token breaking - assume SPACY's tokenizer
    will handle this for us.
    """

    # Indicate section headers, we need them for features
    text = SECTION_HEADER_RE.sub('SECTION-HEADER', text)
    # For simplicity later, remove '.' from most common acronym
    text = text.replace("U.S.", "US")
    text = text.replace('SEC.', 'Section')
    text = text.replace('Sec.', 'Section')
    text = USC_re.sub('USC', text)

    # Remove parantheticals because they are almost always references to laws
    # We could add a special tag, but we just remove for now
    # Note we dont get rid of nested parens because that is a complex re
    # text = PAREN_re.sub('LAWREF', text)
    text = PAREN_re.sub('', text)

    # get rid of HTML tags
    text = HTML_RE.sub('', text)

    # Get rid of enums as bullets or ` as bullets
    text = BULLET_RE.sub(' ', text)

    # Clean html
    text = text.replace('&lt;all&gt;', '')

    # Remove annoying punctuation, that's not relevant
    text = BAD_PUNCT_RE.sub('', text)

    # Get rid of long sequences of dashes - these are formating
    text = DASH_RE.sub(' ', text)

    # removing newlines, tabs, and extra spaces.
    text = WHITESPACE_RE.sub(' ', text)

    # If we ended up with "empty" sentences - get rid of them.
    text = EMPTY_SENT_RE.sub('.', text)

    # Attempt to create sentences from bullets
    text = replace_semicolon(text)

    # Fix weird period issues + start of text weirdness
    # text = re.sub('\.(?=[A-Z])', '  . ', text)
    # Get rid of anything thats not a word from the start of the text
    text = FIX_START_RE.sub('', text)
    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent
    text = FIX_PERIOD.sub(". \g<1>", text)

    # Fix quotes
    text = text.replace('``', '"')
    text = text.replace('\'\'', '"')

    # Add special punct back in
    text = text.replace('SECTION-HEADER', '<SECTION-HEADER>')

    return text


CMU_re = {
    "UNIVERSITY": re.compile('[Cc]\.*[Mm]\.*[Uu]\.*|Carnegie Mellon( University)?|[Tt]he university'),
    "BULLET": re.compile('\n[\ \t]*`*(\(?[a-zA-Z0-9]*\)|[0-9a-zA-Z]*\.)'),
    "HEADER_TBL": re.compile("[A-Z].*[A-Z]:[\n]+[A-Z]+.*[\n]+"),
    "TITLE": re.compile("(POLICY TITLE|Policy Title|TITLE|Title):?[\n]+[A-Z].*[\n]+"),
    "FIX_SENT": re.compile("([a-zA-Z0-9]|\))\s*\n")
}


def find_start_cmu(text):
    # most common section headers for the start of the policy
    start = "\nPolicy Statement\n"
    start2 = "\nStatement\n"

    idx = text.find(start)
    if idx != -1:
        return idx + len(start)
    idx = text.find(start2)
    if idx != -1:
        return idx + len(start2)

    # otherwise, try to find the end of the header table
    tbl_entries = CMU_re["HEADER_TBL"].findall(text)
    if len(tbl_entries) == 0:
        # give up
        return None
    last_entry = tbl_entries[-1]
    return text.find(last_entry) + len(last_entry)


def find_end_cmu(text):
    # all policies in dataset end at the following string
    end = "\nUniversity Policy Office\n"
    return text.find(end)


def get_title_cmu(text):
    match = CMU_re["TITLE"].search(text)
    if match == None:
        return None
    title = text[match.start() + len(match.group(1)) + 1: match.end()]
    title = title.strip()
    return title


def clean_cmu(text):
    start_idx = find_start_cmu(text)
    end_idx = find_end_cmu(text)
    title = get_title_cmu(text)

    if start_idx == None:
        return ""

    text = text[start_idx:end_idx].strip() + "."

    text = "SECTION-HEADER " + text

    # add policy title
    if title != None:
        text = "SECTION-HEADER " + title + ". " + text

    # make sure there is a period before each line break
    text = CMU_re["FIX_SENT"].sub("\g<1>.\n", text)

    # normalize university names
    text = CMU_re["UNIVERSITY"].sub('The University', text)

    # remove bullets
    text = CMU_re["BULLET"].sub(" ", text)

    # Remove annoying punctuation, that's not relevant
    text = BAD_PUNCT_RE.sub('', text)

    # removing newlines, tabs, and extra spaces.
    text = WHITESPACE_RE.sub(' ', text)

    # If we ended up with "empty" sentences - get rid of them.
    text = EMPTY_SENT_RE.sub('.', text)

    # Attempt to create sentences from bullets
    text = replace_semicolon(text)

    # Get rid of anything thats not a word from the start of the text
    text = FIX_START_RE.sub('', text)

    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent
    text = FIX_PERIOD.sub(". \g<1>", text)

    # Fix quotes
    text = text.replace('``', '"')
    text = text.replace('\'\'', '"')

    # Add special punct back in
    text = text.replace('SECTION-HEADER', '<SECTION-HEADER>')

    return text


def clean_cu(text):
    raise Exception("unimplemented")


def clean_dayton(text):
    raise Exception("unimplemented")


def clean_psu(text):
    raise Exception("unimplemented")


def clean_uoregon(text):
    raise Exception("unimplemented")


def split_by_university(data):
    uni_data = {"cmu": pd.DataFrame(columns=['university', 'name', 'policy', 'summary']),
                "cu": pd.DataFrame(columns=['university', 'name', 'policy', 'summary']),
                "dayton": pd.DataFrame(columns=['university', 'name', 'policy', 'summary']),
                "psu": pd.DataFrame(columns=['university', 'name', 'policy', 'summary']),
                "uoregon": pd.DataFrame(columns=['university', 'name', 'policy', 'summary'])}

    for idx, row in data.iterrows():
        uni = row["university"]
        uni_data[uni] = uni_data[uni].append(row, ignore_index=True)

    for uni in uni_data.keys():
        save_path = os.path.join(prefix, "data_uni_sep", uni + ".json")
        uni_data[uni].to_json(save_path, orient='records')


if __name__ == '__main__':

    prefix = os.environ['SUM_DATA']
    path = os.path.join(prefix, 'data_uni_sep')

    # Create dir where data should be saved
    save_path = os.path.join(prefix, 'clean_uni_sep')
    # os.mkdir(save_path)

    for file in os.listdir(path):
        if 'cmu.json' not in file:
            continue

        file_path = os.path.join(path,  file)

        print("Now processing", file_path)

        data = pd.read_json(file_path)

        data['clean_policy'] = data.policy.map(clean_cmu)

        save_path = os.path.join(prefix, 'clean_uni_sep', file)

        data.to_json(save_path, orient='records')

        """data = pd.read_json(file_path, lines=True)[:5]

        data['clean_text'] = data.text.map(clean_text)

        data['clean_summary'] = data.summary.map(clean_text)

        data['clean_title'] = data.title.map(clean_text)

        save_path = os.path.join(prefix, 'billsum_data_clean', file)

        data.to_json(save_path, lines=True, orient='records')"""
