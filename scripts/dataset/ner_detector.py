"""A script to detect named entites in document for use of dataset manually generation."""

from mbaqa import tokenizers
import re


class Detector:
    def __init__(self):
        # init pyltp tool
        annotators = {'ner', 'pos'}
        ltp_tokenizer = tokenizers.get_class('ltp')
        self.tokenizer = ltp_tokenizer(annotators=annotators)
        # regex pattern for dates detection. Need to be improved.
        self.date_pattern = '\d{2,4}(年|月|日|世纪|年代)'

    def first_ne(self, first_index, entity_type, tokens):
        """
        merge consecutive words with the same NER tag to get the first entity phrase
        :param first_index: first index of specific entity in entities(list)
        :param entity_type: Nh(person name)/Ni(organization)/Ns(location)
        :return: phrase <type:str>
        """
        tmp_index = first_index
        result = ''
        # element in entities contains prefix <B/I/E/S>-
        # judge by contains instead of equals
        while tokens.entities()[tmp_index].find(entity_type) != -1:
            result += tokens.words()[tmp_index]
            tmp_index += 1
        return result

    def dates(self, text):
        """
        extract dates in text
        :param text: string
        :return: [date_str1, date_str2, ...]
        """
        return [match.group(0) for match in re.finditer(self.date_pattern, text)]

    def entity_tokens_with_type(self, entity_type, text):
        """
        find all tokens(phrases) of specific entity type, concatenate consecutive tokens
        with the same tag.
        :param text: text string
        :param entity_type: {'Nh', 'Ni', 'Ns'}
        :return: [phrase1, phrase2, ...]
        """
        tokens = self.tokenizer.tokenize(text)
        complete_entity_phrases = []
        tmp_phrase = ''
        for entity, word in zip(tokens.entities(), tokens.words()):
            if entity.find(entity_type) != -1:
                # independent entity
                if 'S' in entity:
                    complete_entity_phrases.append(word)
                # entity begin word
                elif 'B' in entity:
                    complete_entity_phrases.append(tmp_phrase)  # add previous entity
                    tmp_phrase = word  # update tmp phrase
                # entity middle or end word
                elif 'I' in entity or 'E' in entity:
                    tmp_phrase += word
        # last entity
        if len(tmp_phrase) > 0:
            complete_entity_phrases.append(tmp_phrase)
        return complete_entity_phrases

    def contain_date(self, text):
        """detect whether text contains date"""
        return re.search(self.date_pattern, text)

    def contain_person(self, text):
        """detect whether text contain person name"""
        tokens = self.tokenizer.tokenize(text)
        # begin or independent token of Nh(person name) type
        if 'B-Nh' in tokens.entities() or 'S-Nh' in tokens.entities():
            return True
        return False

    def contain_organization(self, text):
        """
        detect whether text contain organization name
        return True and first organization name if text contains organization name
        """
        tokens = self.tokenizer.tokenize(text)
        # begin or independent token of Ni(organization name) type
        if 'B-Ni' in tokens.entities() or 'S-Ni' in tokens.entities():
            return True
        return False

    def contain_location(self, text):
        """
        detect whether text contain location name
        return True and first location if text contains location
        """
        tokens = self.tokenizer.tokenize(text)
        # begin or independent token of Ns(location) type
        if 'B-Ns' in tokens.entities() or 'S-Ns' in tokens.entities():
            return True
        return False



