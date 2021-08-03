###############################################################################
#Dunlin Exceptions
###############################################################################
class DunlinBaseError(Exception):
    @classmethod
    def merge(cls, other, msg=''):
        if other.args:
            new_msg = msg + '\n' + other.args[0]
            e       = cls(new_msg)
        else:
            e = cls(msg)
        try:
            e.num   = other.num
        except:
            pass
        return e
    
    @classmethod
    def merge_raise(cls, other, msg=''):
        new_e = cls.merge(other, msg)
        raise new_e
    
    @classmethod
    def raise_template(cls, details, num):
        e     = cls(details)
        e.num = num
        return e
    