# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#
# This script filters the posts and keeps those posts that are or belong
# to a question that has been asked in 2011 or 2012.
#

import sys
import os
import re
try:
    import ujson as json  # UltraJSON if available
except:
    import json
from dateutil import parser as dateparser

from operator import itemgetter
from xml.etree import cElementTree as etree
from collections import defaultdict

from data import DATA_DIR

#filename = os.path.join(DATA_DIR, "posts-2011-12.xml")
filename = os.path.join(DATA_DIR, "posts-2012.xml")
print("Reading from xml %s" % filename)
filename_filtered = os.path.join(DATA_DIR, "filtered.tsv")
print("Filtered: %s" % filename_filtered)
filename_filtered_meta = os.path.join(DATA_DIR, "filtered-meta.json")
print("Meta: %s" % filename_filtered_meta)

q_creation = {}  # creation datetimes of questions
q_accepted = {}  # id of accepted answer

# question -> [(answer Id, IsAccepted, TimeToAnswer, Score), ...]
meta = defaultdict(list)

# regegx to find code snippets
code_match = re.compile('<pre>(.*?)</pre>', re.MULTILINE | re.DOTALL)
link_match = re.compile(
    '<a href="http://.*?".*?>(.*?)</a>', re.MULTILINE | re.DOTALL)
img_match = re.compile('<img(.*?)/>', re.MULTILINE | re.DOTALL)
tag_match = re.compile('<[^>]*>', re.MULTILINE | re.DOTALL)


def filter_html(s):
    num_code_lines = 0
    link_count_in_code = 0
    code_free_s = s

    num_images = len(img_match.findall(s))

    # remove source code and count how many lines
    for match_str in code_match.findall(s):
        num_code_lines += match_str.count('\n')
        code_free_s = code_match.sub("", code_free_s)

        # sometimes source code contain links, which we don't want to count
        link_count_in_code += len(link_match.findall(match_str))

    links = link_match.findall(s)
    link_count = len(links)

    link_count -= link_count_in_code

    link_free_s = re.sub(
        " +", " ", tag_match.sub('', code_free_s)).replace("\n", "")

    for link in links:
        if link.lower().startswith("http://"):
            link_free_s = link_free_s.replace(link, '')

    num_text_tokens = link_free_s.count(" ")

    return link_free_s, num_text_tokens, num_code_lines, link_count, num_images

years = defaultdict(int)
num_questions = 0
num_answers = 0

if sys.version_info.major < 3:
    # Python 2, map() returns a list, which will lead to out of memory errors.
    # The following import ensures that the script behaves like being executed
    # with Python 3.
    from itertools import imap as map


def parsexml(filename):
    global num_questions, num_answers

    counter = 0

    it = map(itemgetter(1),
             iter(etree.iterparse(filename, events=('start',))))

    root = next(it)  # get posts element

    for elem in it:
        if counter % 100000 == 0:
            print("Processed %i <row/> elements" % counter)

        counter += 1

        if elem.tag == 'row':
            creation_date = dateparser.parse(elem.get('CreationDate'))

            Id = int(elem.get('Id'))
            PostTypeId = int(elem.get('PostTypeId'))
            Score = int(elem.get('Score'))

            if PostTypeId == 1:
                num_questions += 1
                years[creation_date.year] += 1

                ParentId = -1
                TimeToAnswer = 0
                q_creation[Id] = creation_date
                accepted = elem.get('AcceptedAnswerId')
                if accepted:
                    q_accepted[Id] = int(accepted)
                IsAccepted = 0

            elif PostTypeId == 2:
                num_answers += 1

                ParentId = int(elem.get('ParentId'))
                if not ParentId in q_creation:
                    # question was too far in the past
                    continue

                TimeToAnswer = (creation_date - q_creation[ParentId]).seconds

                if ParentId in q_accepted:
                    IsAccepted = int(q_accepted[ParentId] == Id)
                else:
                    IsAccepted = 0

                meta[ParentId].append((Id, IsAccepted, TimeToAnswer, Score))

            else:
                continue

            Text, NumTextTokens, NumCodeLines, LinkCount, NumImages = filter_html(
                elem.get('Body'))

            values = (Id, ParentId,
                      IsAccepted,
                      TimeToAnswer, Score,
                      Text.encode("utf-8"),
                      NumTextTokens, NumCodeLines, LinkCount, NumImages)

            yield values

            root.clear()  # preserve memory

with open(filename_filtered, "w") as f:
    for values in parsexml(filename):
        line = "\t".join(map(str, values))
        f.write(line + "\n")

with open(filename_filtered_meta, "w") as f:
    json.dump(meta, f)

print("years:", years)
print("#qestions: %i" % num_questions)
print("#answers: %i" % num_answers)
