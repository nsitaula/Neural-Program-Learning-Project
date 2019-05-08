Package required:
* Tensorflow
* Tflearn
* Numpy (if not already installed)
* Pickle (if not already installed)

You can use NPI_Country_Region.ipynb file and having all other files in the same folder except for the separate folder for model named as model.
The cells header of the .ipynb file explains the use of the cell i.e. generate data, train, plot, evaluate or test.

Alternatively, the files description can be as follows:
create_program_data -> for creating the trace of the data
train.py -> for training the data
eval.py -> for testing or evaluating of the data on randomly picked data

Note that to continue press “c” while running the eval.py or testing/evaluation cell in .ipynb file.

Example of trace from the cell would be:
To find:  Angola

--------  Environment ----------
>  Angola   None
>  Nepal   Asia
>  India   Asia
>  China   Asia
>  Peru   North America
>  Brazil   North America
>  UK   Europe
>  Sudan   Africa
>  Italy   Europe
>  Sweden   Europe
>  Mexico   North America
>  USA   North America
>  Angola   Africa
>  Bangladesh   Asia
>  Japan   Asia
>  Egypt   Africa
>  Nigeria   Africa
>  None   None

-----------------------------------
Step: FIND, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Continue? c
Step: INCREMENT, Arguments: [], Terminate: False
Step: INCREMENT, Arguments: [], Terminate: True
>  Angola   None
>  Nepal   Asia
>  India   Asia
>  China   Asia
>  Peru   North America
>  Brazil   North America
>  UK   Europe
>  Sudan   Africa
>  Italy   Europe
>  Sweden   Europe
>  Mexico   North America
>  USA   North America
>  Angola   Africa
>  Bangladesh   Asia
>  Japan   Asia
>  Egypt   Africa
>  Nigeria   Africa
>  None   None

Pointer at  12
-------- KB Environment ----------
>  Angola   None
>  Nepal   Asia
>  India   Asia
>  China   Asia
>  Peru   North America
>  Brazil   North America
>  UK   Europe
>  Sudan   Africa
>  Italy   Europe
>  Sweden   Europe
>  Mexico   North America
>  USA   North America
>  Angola   Africa
>  Bangladesh   Asia
>  Japan   Asia
>  Egypt   Africa
>  Nigeria   Africa
>  Africa   None

--------------------------------------------------------------------------
Model Output:  Africa
Correct Out :  Africa
Correct!

 



