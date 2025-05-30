# Blogging with Jupyter Notebooks for AI development

#### Testing, testing... Is this thing on?

```python
print('Hello, World!')
```

    Hello, World!
    

#### Hey Siri, what is 1+1?


```python
1+1
```




    2



#### Hey Siri, create a data frame and a line plot.


```python
import matplotlib.pyplot as plt
import pandas as pd
```


```python
pd.DataFrame({'a':[1,2], 'b':[3,4]})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot([1,2])
```




    [<matplotlib.lines.Line2D at 0x1ce8b9a0a70>]




<img src="images/output_8_1.png" height="300">


#### Hey Siri, show me a picture of a robot.


<img src="images/HONDA_ASIMO.jpg" height="300">

