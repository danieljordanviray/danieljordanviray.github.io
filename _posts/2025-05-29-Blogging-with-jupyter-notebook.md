# Blogging with Jupyter Notebooks for AI development


```python
# hide
# pip install nbdev
```


```python
# hide
print('Hello, World!')
```

    Hello, World!
    

#### AI, what is 1+1?


```python
1+1
```




    2



#### AI, create a data frame and a line plot


```python
# hide
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# hide
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
# hide
plt.plot([1,2])
```




    [<matplotlib.lines.Line2D at 0x1ce8b9a0a70>]




    
![](/images/output_8_1.png)
    


#### AI, show me a picture of a robot.


```python
# hide
```

![](/images/HONDA_ASIMO.jpg)

