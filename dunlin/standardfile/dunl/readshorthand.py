from collections import UserDict

import dunlin.standardfile.dunl.readstring as rst
import dunlin.standardfile.dunl.delim as dm

###############################################################################
# Substitution of Shorthands
###############################################################################
def substitute(template, horizontals, verticals):
    temp = substitute_horizontals(template, horizontals)
    result = substitute_verticals(temp, verticals)

    return result


def substitute_horizontals(template, horizontals):
    parsed_horizontals = {}
    for key, dct in horizontals.items():
        temp = zip_substitute(dct.template, dct)
        temp = dct.join.join(temp)

        parsed_horizontals[key] = temp

    result = format_string(template, **parsed_horizontals)

    return result


def substitute_verticals(template, verticals):
    return zip_substitute(template, verticals, max_brackets=1)


def zip_substitute(template, dct, max_brackets=2):
    if not dct:
        return [format_string(template)]

    result = []
    keys = dct.keys()
    for row in zip(*dct.values()):
        local = dict(zip(keys, row))
        temp = format_string(template, max_brackets, **local)
        result.append(temp)

    return result


def read_shorthand(element):
    temp = get_template_and_shorthands(element)
    return substitute(*temp)


def format_string(string, max_brackets=2, /, **kwargs):
    result = []
    curr = ''
    quotes = []
    for i, char in enumerate(string):
        if char in dm.quotes:
            if not quotes:
                if curr:
                    result.append(curr)

                quotes.append(char)
                curr = char

            elif char == quotes[-1]:
                curr += char
                quotes.pop()
                result.append(curr)
                curr = ''

            else:
                curr += char
                quotes.append(char)
        else:
            curr += char

    result.append(curr)

    # Perform substitution
    for i in range(len(result)):
        temp = result[i]
        if not temp:
            continue
        elif temp[0] in dm.quotes:
            continue

        nospace = ''.join(temp.split())
        if '{'*(max_brackets+1) in nospace or '}'*(max_brackets+1) in nospace:
            msg = f'Max. ({max_brackets+1}) number of levels of curly braces exceeded.'
            raise SyntaxError(msg)
        else:
            try:
                result[i] = temp.format(**kwargs)
            except KeyError as e:
                msg = f'Unexpected field: {e.args[0]}'
                
                raise ValueError(msg)
            except Exception as e:
                raise e

    # Join and return a string
    result = ''.join(result)
    return result

###############################################################################
# Splitting of Element
###############################################################################
def get_template_and_shorthands(element):

    template, found = string2chunks(element)
    vertical = {}
    horizontal = {}

    for shorthand_type, key, value in found:
        key = key.strip()

        check_valid_shorthand_key(shorthand_type, key, value)

        if shorthand_type == dm.vertical:
            values_ = rst.split_top_delimiter(value)
            if not values_:
                raise ValueError(
                    f'Shorthand must have at least one value.\n{element}')

            vertical[key] = values_

        elif shorthand_type == dm.horizontal:
            split = key.split('.')
            if len(split) == 1:
                horizontal.setdefault(key, Horizontal())
                horizontal[key].template = value.strip()
            else:
                key, attr = split
                key = key.strip()
                attr = attr.strip()
                horizontal.setdefault(key, Horizontal())

                if attr == 'join' or attr == '_j':
                    value = value.strip()
                    if value[0] == value[-1] and value[0] in dm.quotes:
                        join = value[1:-1]
                    else:
                        join = value

                    horizontal[key].join = join
                else:
                    values_ = rst.split_top_delimiter(value, ',')
                    # values_ = rst.split_top_delimiter(value, ',')
                    if not values_:
                        raise ValueError(
                            f'Shorthand must have at least one value.\n{element}')

                    horizontal[key][attr] = values_
        else:
            raise ValueError(
                f'No shorthand type "{shorthand_type}"\n{element}')
    return template, horizontal, vertical


def check_valid_shorthand_key(shorthand_type, key, value):
    if not value:
        msg = 'No value associated with the following shorthand.'
        msg += ' It may be missing a delimiter.'
        msg += f'\n{(shorthand_type, key, value)}'
        raise ValueError(msg)


def string2chunks(string):
    template = None
    i0 = 0
    quote = []
    delimiter = ''
    key = None
    value = None
    chunks = []

    for i, char in enumerate(string):

        if char == dm.horizontal and not quote:
            if template is None:
                template = string[i0:i]

            if key is None and value is None:
                delimiter += char
                i0 = i + 1
                continue
            elif key is not None and value is not None:
                chunks.append([delimiter, key, value])
                delimiter = char
                key = None
                value = None
                i0 = i + 1
                continue
            else:
                msg0 = f'Missing or incomplete key-value pair in {string}.'
                msg1 = f'Check {string[i-10: i+10]}'
                msg = msg0 + '\n' + msg1
                raise ValueError(msg)

        elif char == dm.pair and not quote and template is not None and value is None:
            value = ''
            continue

        elif char in dm.quotes:
            if not quote:
                quote.append(char)
            elif quote[-1] == char:
                quote.pop()
            else:
                quote.append(char)

        if template is not None:
            if key is None:
                key = char
            elif value is None:
                key += char
            else:
                value += char

    if template is None:
        return string, []

    elif delimiter:
        chunks.append([delimiter, key, value])

    return template, chunks


class Horizontal(UserDict):
    template = ''
    join = ', '
