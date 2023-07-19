# # Description
The `datastructures` submodule processes and checks raw input and instantiates classes corresponding to the various data structures used by Dunlin for instantiating models.


# Motivation
In principle, it is possible to write a program such that models are directly instantiated from raw Python data i.e. dicts, lists, strings etc. However, this would be a bad idea for the following reasons:

1. Each data structure—reactions for example—have highly specific requirements regarding their content and formatting.
2. Data structures also need to cross-checked with one another. For example, states that appear in reactions cannot appear in rates and vice versa. 
3. The the large size of a model means that users could easily make mistakes when typing out a model.
   1. Dunlin must be able to direct the user to the exact location of a mistake.
   2. Dunlin must also allow the developer to differentiate between bad input and internal programming errors.
4. Additional methods/functions need to be coded to allow round-trips back to the input as well as exporting into other formats.

All of this requires extensive checking and formatting and the use of auxiliary functionalities that have nothing to do with the actual modeling process. So much so that it is neater to place them in their own submodule. 

# Implementation
A model in Dunlin will consists of a multitude of data structures. These correspond to states, parameters, reactions and more. If a data structure requires a specific format and is considered important to the functioning of the model, a class should be created specifically to process the raw input. If a data structure does not meet these two criteria, it can simply be deep-copied from the user input.

The data structures are to be contained as attributes within another class corresponding to the model data. The class for the model data must implement the instantiation of the data structures. It should also contain methods for exporting the data structures back in to raw Python or other formats such as Dunlin code. 

All this means that a developer working downstream will instantiate the model data from the raw input and then extract information from the model data to generate the code/functions required for modeling.

The flow of information is thus as follows:

`Raw data <-> Model data containing data structures <-> Model for simulation and analysis`

Note: To streamline the implementation of each class, the submodule contains base classes which serve a templates.

