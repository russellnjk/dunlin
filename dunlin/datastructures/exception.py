class InvalidDefinition(Exception):
    def __init__(self, structuretype, expected='', received=''):
        msg = f'Invalid definition of {structuretype}.'
        
        if expected:
            msg += f'\nExpected: {expected}'
        if received:
            msg += f'\nReceived: {received}'
            
        super().__init__(msg)