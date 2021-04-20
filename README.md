# ResponseEncoding
Encode Categorical Features based on Target/Class

**Example:**
~~~
> re = ResponseEncoder()
> re.fit(X_train[feature],y)
> encoded_feature= re.transform(X_test[feature])
~~~
