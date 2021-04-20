class ResponseEncoder:
    """
    Converts Categorical feature into numericals as per y/Target/Class/Dependent Feature.
    Example:- 
    > re = ResponseEncoder()
    > re.fit(X_train[feature],y)
    > encoded_feature= re.transform(X_test[feature])
    """
    def __init__(self):
        self.table= None
        self.category=None
        self.target=None
    
    def make_table(self,X,y):
        """Returns [p(Y=1|X),p(Y=2|X)...p(Y=n|X)] for each unique i in X"""
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        category = list(set(X))
        target = list(set(y))

        table = np.zeros((len(category),len(target)))

        for i in range(len(X)):
            index1 = category.index(X[i])
            index2 = target.index(y[i])
            table[index1,index2] += 1

        table /= table.sum(axis=1).reshape(-1,1)

        return table,category,target

    def encode_using_table(self,X):
        """Returns [p(Y=1|X),p(Y=2|X)...p(Y=n|X)] for each i in X"""
        X.reset_index(drop=True, inplace=True)
        
        xtable = np.zeros((len(X),len(self.target)))
        
        for i in range(len(X)):
            if X[i] in self.category:
                xtable[i,:] = self.table[self.category.index(X[i]),:]
            else:
                xtable[i,:] = (1/len(self.target))*np.ones((len(self.target)))

        return xtable
    
    def fit(self,X,y):
        assert len(X) == len(y), "Both Inputs must be equal in size"
        
        self.table, self.category, self.target = self.make_table(X,y)
    
    def transform(self, X):
        return self.encode_using_table(X)
