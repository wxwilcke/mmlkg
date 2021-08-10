#! /usr/bin/env python


class Statement(tuple):
    """Statement"""

    subject = None
    predicate = None
    object = None

    def __new__(cls, subject, predicate, object):
        return super().__new__(cls, (subject, predicate, object))

    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __getnewargs__(self):
        return (self.subject, self.predicate, self.object)

    def __eq__(self, other):
        for resourceA, resourceB in ((self.subject, other.subject),
                                     (self.predicate, other.predicate),
                                     (self.object, other.object)):
            if resourceA != resourceB:
                return False

        return True

    def __lt__(self, other):
        # ordering following predicate logic: (s, p, o) := p(s, o)
        for resourceA, resourceB in ((self.predicate, other.predicate),
                                     (self.subject, other.subject),
                                     (self.object, other.object)):
            if resourceA < resourceB:
                return True

        return False

    def __str__(self):
        return "(%s, %s, %s)" % (str(self.subject),
                                 str(self.predicate),
                                 str(self.object))

    def __hash__(self):
        return hash(repr(self))


class Resource:
    value = None

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def __hash__(self):
        return hash(repr(self))


class Literal(Resource):
    datatype = None
    language = None

    def __init__(self, value, datatype=None, language=None):
        super().__init__(value)

        if datatype is not None and language is not None:
            raise Warning("Accepts either datatype or language, not both")

        self.datatype = IRIRef(datatype) if datatype is not None else None
        self.language = language

    def __eq__(self, other):
        return self.value == other.value\
                and self.datatype == other.datatype\
                and self.language == other.language

    def __hash__(self):
        value = str()
        if self.datatype is not None:
            value = self.datatype
        if self.language is not None:
            value = self.language

        return hash(repr(self)+repr(value))


class BNode(Resource):
    def __init__(self, value):
        super().__init__(value)


class IRIRef(Resource):
    def __init__(self, value):
        super().__init__(value)


def parse_statement(statement):
    subject = _parse_subject(statement[0])
    predicate = _parse_predicate(statement[1])
    object = _parse_object(statement[2])

    return Statement(subject, predicate, object)


def _parse_subject(node):
    if node.startswith("_:"):
        return BNode(node[2:])
    else:  # iriref
        return IRIRef(node)


def _parse_predicate(predicate):
    return IRIRef(predicate)


def _parse_object(node):
    if not node.startswith('\"'):
        if node.startswith("_:"):
            return BNode(node[2:])
        else:  # iriref
            return IRIRef(node)

    language = None
    datatype = None
    if node.endswith('>'):
        # datatype declaration
        for i in range(len(node)):
            if node[-i] == '<':
                break

        datatype = node[-i+1:-1]
        node = node[:-i-2]  # omit ^^
    elif not node.endswith('\"'):
        # language tag
        for i in range(len(node)):
            if node[-i] == '@':
                break

        language = node[-i+1:]  # omit @-part
        node = node[:-i]
    elif node.startswith('\"') and node.endswith('\"'):
        pass
    else:
        raise Exception("Unsuspected format: " + node)

    return Literal(node, language=language, datatype=datatype)
