from causalgraphicalmodels import CausalGraphicalModel
import matplotlib.pyplot as plt

sprinkler = CausalGraphicalModel(
    nodes=["season", "rain", "sprinkler", "wet", "slippery"],
    edges=[
        ("season", "rain"),
        ("season", "sprinkler"),
        ("rain", "wet"),
        ("sprinkler", "wet"),
        ("wet", "slippery")
    ]
)

# draw return a graphviz `dot` object, which jupyter can render
sprinkler.draw()

# get the distribution implied by the graph
print(sprinkler.get_distribution())

# get all the conditional independence relationships implied by a CGM
sprinkler.get_all_independence_relationships()

# check backdoor adjustment set
sprinkler.is_valid_backdoor_adjustment_set("rain", "slippery", {"wet"})

frozenset({frozenset({'sprinkler'}),
           frozenset({'season'}),
           frozenset({'season', 'sprinkler'})})

# get the graph created by intervening on node "rain"
do_sprinkler = sprinkler.do("rain")

do_sprinkler.draw()

dag_with_latent_variables = CausalGraphicalModel(
    nodes=["x", "y", "z"],
    edges=[
        ("x", "z"),
        ("z", "y"),
    ],
    latent_edges=[
        ("x", "y")
    ]
)

dag_with_latent_variables.draw()

# here there are no observed backdoor adjustment sets
dag_with_latent_variables.get_all_backdoor_adjustment_sets("x", "y")

dag_with_latent_variables.get_all_backdoor_adjustment_sets("x", "y")
#%%
# but there is a frontdoor adjustment set
dag_with_latent_variables.get_all_frontdoor_adjustment_sets("x", "y")

"""
#%% md
# StructuralCausalModels

For Structural Causal Models (SCM) we need to specify the functional form of each node:

"""

from causalgraphicalmodels import StructuralCausalModel
import numpy as np

scm = StructuralCausalModel({
    "x1": lambda     n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
    "x2": lambda x1, n_samples: np.random.normal(loc=x1, scale=0.1),
    "x3": lambda x2, n_samples: x2 ** 2,
})

"""
#%% md
The only requirement on the functions are:
 - that variable names are consistent 
 - each function accepts keyword variables in the form of `numpy` arrays and output numpy arrays of shape [n_samples] 
 - that in addition to it's parents, each function takes a `n_samples` variables indicating how many samples to generate 
 - that any function acts on each row independently. This ensure that the output samples are independent
 
Wrapping these functions in the `StructuralCausalModel` object allows us to easily generate samples: 

"""

ds = scm.sample(n_samples=100)

df = ds.head()
print(df.to_string())

# and visualise the samples
import seaborn as sns

sns.kdeplot(
    x=ds.x2,
    y=ds.x3
)
plt.show()

scm.cgm.draw()

scm_do = scm.do("x1")

scm_do.cgm.draw()

df = scm_do.sample(n_samples=5, set_values={"x1": np.arange(5)})

print(df.to_string())
do_break = 1


