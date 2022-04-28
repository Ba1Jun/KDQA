import json
import six
import subprocess
import tokenization
import logging
import torch
import pandas as pd
import collections
from collections import OrderedDict
from torch.utils.data import TensorDataset
import os
import operator


class SquadExample(object):
    """A single training/test example for simple sequence classification.
     For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s\n" % (self.qas_id)
        s += ", question_text: %s\n" % (self.question_text)
        s += ", answer_text: [%s]\n" % (" ".join(self.orig_answer_text))
        s += ", doc_tokens: [%s]\n" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d\n" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d\n" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r\n" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 attention_mask,
                 query_mask,
                 passage_mask,
                 ans_mask,
                 token_type_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.query_mask = query_mask
        self.passage_mask = passage_mask
        self.ans_mask = ans_mask
        self.token_type_ids = token_type_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_squad_examples(input_file, is_training, is_bioasq=True):
    """Read a SQuAD json file into a list of SquadExample."""

    with open(input_file, "r") as reader:
        # if is_bioasq:
        # input_data = [{u'paragraphs':json.load(reader)["questions"], u'title':'bioASQ'}] # to fit the shape of squad code
        # else:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            if is_bioasq:
                paragraph_text.replace('/', ' ')  # need review
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    # if (len(qa["answers"]) != 1) and (not is_impossible):
                    #     raise ValueError("For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            print("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def squad_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                       doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            token_type_ids = []
            tokens.append("[CLS]")
            token_type_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                token_type_ids.append(0)
            tokens.append("[SEP]")
            token_type_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                token_type_ids.append(1)
            tokens.append("[SEP]")
            token_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                token_type_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(token_type_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0
            
            # if is_training and start_position == 0 and end_position == 0:
            #     continue
            
            if start_position is None:
                query_mask = attention_mask
                passage_mask = attention_mask
                ans_mask = attention_mask
            else:
                query_mask = [0] + [1] * len(query_tokens)
                while len(query_mask) < max_seq_length:
                    query_mask.append(0)
                
                passage_mask = [0] * (len(query_tokens)+2) + [1] * doc_span.length
                while len(passage_mask) < max_seq_length:
                    passage_mask.append(0)
                
                ans_mask = [0] * start_position + [1] * (end_position-start_position+1)
                while len(ans_mask) < max_seq_length:
                    ans_mask.append(0)


            if example_index < 20:
                logging.info("*** Example ***")
                logging.info("unique_id: %s" % (unique_id))
                logging.info("example_index: %s" % (example_index))
                logging.info("doc_span_index: %s" % (doc_span_index))
                logging.info("tokens: %s" % " ".join(
                    [x for x in tokens]))
                logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info(
                    "attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logging.info(
                    "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                if is_training and example.is_impossible:
                    logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logging.info("start_position: %d" % (start_position))
                    logging.info("end_position: %d" % (end_position))
                    logging.info(
                        "answer: %s" % (answer_text))

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                query_mask=query_mask,
                passage_mask=passage_mask,
                ans_mask=ans_mask,
                token_type_ids=token_type_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)

            features.append(feature)

            unique_id += 1

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_query_masks = torch.tensor([f.query_mask for f in features], dtype=torch.long)
    all_passage_masks = torch.tensor([f.passage_mask for f in features], dtype=torch.long)
    all_ans_masks = torch.tensor([f.ans_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

    if not is_training:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_query_masks, all_passage_masks
        )
    else:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_positions,
            all_end_positions,
            all_is_impossible,
            all_query_masks,
            all_passage_masks,
            all_ans_masks,
            all_feature_index
        )

    return features, dataset


def textrip(text):
    if text=="":
        return text
    if text[-1]==',' or text[-1]=='.' or text[-1]==' ':
        return text[:-1]
    if len(text)>2 and text[0]=='(' and text[-1]==')':
        if text.count('(')==1 and text.count(')')==1:
            return text[1:-1]
    if ('(' in text) and (')' not in text):
        return ""
    if ('(' not in text) and (')' in text):
        return ""
    return text


def transform_n2b_factoid(nbest_path, output_path):
    #### Checking nbest_BioASQ-test prediction.json
    if not os.path.exists(nbest_path):
        print("No file exists!\n#### Fatal Error : Abort!")
        raise

    #### Reading Pred File
    with open(nbest_path, "r") as reader:
        test = json.load(reader)

    qidDict = dict()
    if True:
        for multiQid in test:
            assert len(multiQid) == (24 + 4)  # all multiQid should have length of 24 + 3
            if not multiQid[:-4] in qidDict:
                qidDict[multiQid[:-4]] = [test[multiQid]]
            else:
                qidDict[multiQid[:-4]].append(test[multiQid])
    else:  # single output
        qidDict = {qid: [test[qid]] for qid in test}

    entryList = []
    entryListWithProb = []
    for qid in qidDict:

        jsonList = []
        for jsonele in qidDict[qid]:  # value of qidDict is a list
            jsonList += jsonele

        qidDf = pd.DataFrame().from_dict(jsonList)

        sortedDf = qidDf.sort_values(by='probability', axis=0, ascending=False)

        sortedSumDict = OrderedDict()
        sortedSumDictKeyDict = dict()  # key : noramlized key

        for index in sortedDf.index:
            text = sortedDf.loc[index]["text"]
            text = textrip(text)
            if text == "":
                pass
            elif len(text) > 100:
                pass
            elif text.lower() in sortedSumDictKeyDict:
                sortedSumDict[sortedSumDictKeyDict[text.lower()]] += sortedDf.loc[index]["probability"]
            else:
                sortedSumDictKeyDict[text.lower()] = text
                sortedSumDict[sortedSumDictKeyDict[text.lower()]] = sortedDf.loc[index]["probability"]
        finalSorted = sorted(sortedSumDict.items(), key=operator.itemgetter(1),
                             reverse=True)  # for python 2, use sortedSumDict.iteritems() instead of sortedSumDict.items()

        entry = {u"type": "factoid",
                 # u"body":qas,
                 u"id": qid,  # must be 24 char
                 u"ideal_answer": ["Dummy"],
                 u"exact_answer": [[ans[0]] for ans in finalSorted[:5]],
                 # I think enough?
                 }
        entryList.append(entry)

        entryWithProb = {u"type": "factoid",
                         u"id": qid,  # must be 24 char
                         u"ideal_answer": ["Dummy"],
                         u"exact_answer": [ans for ans in finalSorted[:20]],
                         }
        entryListWithProb.append(entryWithProb)
    finalformat = {u'questions': entryList}
    finalformatWithProb = {u'questions': entryListWithProb}

    if os.path.isdir(output_path):
        outfilepath = os.path.join(output_path, "BioASQform_BioASQ-answer.json")
        outWithProbfilepath = os.path.join(output_path, "WithProb_BioASQform_BioASQ-answer.json")
    else:
        outfilepath = output_path
        outWithProbfilepath = output_path + "_WithProb"

    with open(outfilepath, "w") as outfile:
        json.dump(finalformat, outfile, indent=2)
    with open(outWithProbfilepath, "w") as outfile_prob:
        json.dump(finalformatWithProb, outfile_prob, indent=2)


def eval_bioasq_standard(task_num, outfile, golden, cwd):
    # 1: [1, 2],  3: [3, 4],  5: [5, 6, 7, 8]

    task_e = {
        '1': 1, '2': 1,
        '3': 3, '4': 3,
        '5': 5, '6': 5, '7': 5, '8': 5
    }

    golden = os.path.join(os.getcwd(), golden)
    outfile = os.path.join(os.getcwd(), outfile)

    evalproc1 = subprocess.Popen(
        ['java', '-Xmx10G', '-cp',
         '$CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar',
         'evaluation.EvaluatorTask1b', '-phaseB',
         '-e', '{}'.format(task_e[task_num]),
         golden,
         outfile],
        cwd=cwd,
        stdout=subprocess.PIPE
    )
    stdout1, _ = evalproc1.communicate()

    result = [float(v) for v in stdout1.decode('utf-8').split(' ')]

    return result
