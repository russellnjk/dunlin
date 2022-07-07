import addpath
import dunlin as dn
import dunlin.standardfile.dunl.readdunl as rdun

#Test with filename
filename = 'test_load.dunl'
all_data = rdun.read_file(filename)

assert len(all_data) == 1
assert 'M1' in all_data

m1 = all_data['M1']

assert m1 == {'a' : 1}
    
print(m1)

with open(filename, 'r') as file:
    all_data = rdun.read_file(file)

assert len(all_data) == 1
assert 'M1' in all_data

m1 = all_data['M1']

assert m1 == {'a' : 1}
    
print(m1)