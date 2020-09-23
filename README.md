# DownSemble

> In many real-world problems, data groups are imbalanced, so that most of the population belongs to one group and only a minority to another group.

> The library was created for the Jupyter notebook

> The main problem with unbalanced data is that for the most part, the model will tend to say about an instance that it belongs to the majority population, simply because statistically there is such a greater chance.

> This classifier is designed to handle unbalanced data. The classification is based on an ensemble of sub-sets.

        Input:

            base_estimator     A base model that the classifier will use to make a prediction on each subset.

            ratio              The ratio of the minority group to the rest of the data. 
                               The default is 1.
                               The ratio describes a ratio of 1: ratio.
                               For example:
                               Ratio = 1 Creates a 1: 1 ratio (50% / 50%) between the minority group and the rest of the data.
                               Ratio = 2 Creates a 2: 1 ratio (33% / 66%) between the minority group and the rest of the data.

            ensemble           The method by which the models of each subset will be combined together. 
                               The default is mean.
                               For numeric labels you can select max or min to tilt the classifier to a certain side. 

            random_state       Seed for the distribution of the majority population in each subset.
                                The default is 42.

        Attributes:
        
            fit(X_train, y_train)
            
            predict(X)
            
            predict_proba(X)
            
            list_of_df         List of all created sub-sets.
            
            list_models        List of all the models that make up the final model.
			

Roni Gold
ronigoldsmid@gmail.com