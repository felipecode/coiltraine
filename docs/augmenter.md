
The augmenter module is a mixture betweent many:

imgaug
pytorch vision augmentation.


THe augmentar is a sequence of callables.
It is kind of hard to control it on a config file. 
Maybe with some kind of dictionary structure ?

However it also should be quite able to change the parameters

THe best option is to have an augmentation profile created , As as separate
config file, and you could just choose this profile of augmentation
