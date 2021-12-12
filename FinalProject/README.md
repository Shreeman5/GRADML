This README will explain how to get the results:

Go to lines 477-479. Uncomment these lines. These lines were commented out because they served their purpose of finding the best depth for both Entropy and Gini Index.
Now go to the method find_best_depth in line 361. This method gives the average accuracy for every depth. You can change the depth of the decision trees at line 362.

Now, to get results(which I documented in the final report) for the Entropy Model, do this:

Specify depth = 1, Run the Program, Verify the results
Specify depth = 2, Run the Program, Verify the results
..............................
...............................
Specify depth = 13, Run the Program, Verify the results

To get results(which I documented in the final report) for the Gini Index model, do this:

Comment out lines 367-375
Uncomment lines 376-384

Specify depth = 1, Run the Program, Verify the results
Specify depth = 2, Run the Program, Verify the results
..............................
...............................
Specify depth = 13, Run the Program, Verify the results


As you can see, my program will show that depth = 2 is the best hyperparameter for both models.

Comment out lines 477-479.

Look at lines 481 and 482. The 2 in those lines simply is depth. Now you can understand why I commented out lines 477-479 in the first place. 

Those trees are then returned to the testing function in line 490. But first, uncomment line 535.


To generate a csv file for the Entropy model, simply run the program. Check for accuracy in Kaggle.

To generate a csv file for the Gini Index model, uncomment lines 514-521 and comment out lines 523-30. Then run the program. Check for accuracy in Kaggle.



