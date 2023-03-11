import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import operator
import random
from PIL import Image

#image = Image.open('D:\Codes\Python\Club\streamlit-app-knapsack-main\streamlit-app-knapsack-main\png1.png')
#####################################################################################################
#Some Markdown
st.title('Traveling Salesman Solver')

st.write(""" The travelling salesman problem (also called the travelling salesperson problem or TSP) asks the following question: "Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?" It is an NP-hard problem in combinatorial optimization, important in theoretical computer science and operations research.""")
#####################################################################################################
#Importing Data
df1 = pd.read_json('D:\Codes\Python\Club\Genetic-Algorithm-for-TSP-resolution-main\coordinates.json')
df2 = df1[0:15]
df3 = df1[0:15]
df3.drop('lat',inplace=True,axis=1)
df3.drop('lng',inplace=True,axis=1)
#####################################################################################################
#Fixing parameters
nbr_villes=len(df2) 
population_taille=30
#####################################################################################################
#Getting coordinates
x = df2['lng'].to_numpy()
y = df2['lat'].to_numpy()
cityList = []
for i in range(0,nbr_villes):
    l = [x[i]]
    l.append(y[i])
    cityList.append(l)
#####################################################################################################
#Algorithm functions
def creer_nv_individu(n_villes):

    pop=set(np.arange(n_villes,dtype=int))
    route=list(random.sample(list(pop),n_villes)) #route est un individu aléatoirement générer
    for i in range(len(route)):
        if route[i] == 0:
            route = np.roll(route, -i, axis=None)# np.roll inverse les tournées pour mettre la ville 0 (départ) toujours en premier sans changer la solution
            
    return route
def cree_population_initial(taille,n_villes):
    population = []
    
    for i in range(0,taille):
        population.append(creer_nv_individu(n_villes))
        
    return population
def distance(i,j):
    return np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
def fitness(route,CityList):
    score=0    
    for i in range(len(route)-1):
        k=route[i] 
        l=route[i+1]
        score = score + distance(CityList[k],CityList[l])            
    return score
def score_population(population, CityList):  
    scores = []  
    for i in population:
        scores.append(fitness(i, CityList))
    return scores
def selectOne(population):   # séléction d'un individu
    total = sum([fitness(c,cityList) for c in population])

    selection_prob = [fitness(c,cityList)/total for c in population] #proba de maximisation
    selection_probs = [(1-selection_prob[i])/(population_taille-1) for i in range(population_taille)] #proba de minimisation
    return population[np.random.choice(len(population), p=selection_probs)]  #séléction par roullette (proba aléatoire)
def selection(population):   # séléction de N/2 individus dans la population
    selected = []
    for i in range(int(len(population)/2)):
        selected.append(selectOne(population))
    return selected
def crossover(a,b): 
    child=[]
    childA=[]
    childB=[]
    geneA=int(random.random()* len(a))
    geneB=int(random.random()* len(b))    
    start_gene=min(geneA,geneB)
    end_gene=max(geneA,geneB)    
    for i in range(start_gene,end_gene):
        childA.append(a[i])        
    childB=[item for item in a if item not in childA]
    child=childA+childB   
    return child
def breedPopulation(selected):
    children=[]
    for i in range(len(selected)-1):
        children.append(crossover(selected[i],selected[i+1]))
    return children
def mutate(route,probablity):
    route=np.array(route)
    for swaping_p in range(len(route)):
        if(random.random() < probablity):
            swapedWith = np.random.randint(0,len(route))          
            temp1=route[swaping_p]          
            temp2=route[swapedWith]
            route[swapedWith]=temp1
            route[swaping_p]=temp2   
    return route
def mutatePopulation(population, mutationRate):
    mutatedPop = []   
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop
def rankRoutes(population,City_List):
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = fitness(population[i],City_List)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False)
def best_solution(population,city_List):
    ranked_population = rankRoutes(population,city_List)
    i = ranked_population[0][0]
    solBest = population[i]
    return solBest
def nextGeneration(CityList,currentGen, mutationRate):
    selectionResults = selection(currentGen)
    children = breedPopulation(selectionResults)
    nextGeneration = mutatePopulation(children, mutationRate)
    nextGeneration = nextGeneration + selectionResults 
    nextGeneration.append(best_solution(currentGen,CityList))
    return nextGeneration
#####################################################################################################
population_initiale = cree_population_initial(population_taille,nbr_villes)
#####################################################################################################
#fixing distances to km
coef =1/(distance(cityList[0],cityList[1])/351.21)
#####################################################################################################
def geneticAlgorithm(CityList,mutationRate,generations):   
    bestfitness_pergen = []
    gen = []
    population = population_initiale
    # liste des meilleurs distances par génération
    bestfitness_pergen.append(fitness(best_solution(population_initiale,CityList),CityList))
    for i in range(0, generations):
        population = nextGeneration(CityList,population, mutationRate)
        bestfitness_pergen.append(fitness(best_solution(population,CityList),CityList))
        gen.append(i+1)   
    # meilleur solution de la dernière population    
    route = best_solution(population,CityList)
    for i in range(len(route)):
        if route[i] == 0:
            route = np.roll(route, -i, axis=None)
    return route
#####################################################################################################
st.write(""" ##### Your challenge here is to pick the shortest route starting from Algiers and traveling to all the wilayas given on the list and then back to Algiers""")
#####################################################################################################
#Visualization
view_state = pdk.ViewState(
    latitude=32.4833,
    longitude=3.6667,
    zoom=3.5
)
layer =[pdk.Layer(
    "ScatterplotLayer",
    df1,
    pickable=True,
    opacity=1,
    stroked=True,
    filled=True,
    radius_scale=10,
    radius_min_pixels=5,
    radius_max_pixels=100,
    line_width_min_pixels=0,
    get_position="coordinates",
    get_fill_color=[255, 140, 0],
    get_line_color=[0, 0, 0],
),
pdk.Layer(
    "ScatterplotLayer",
    df2,
    pickable=True,
    opacity=1,
    stroked=True,
    filled=True,
    radius_scale=10,
    radius_min_pixels=5,
    radius_max_pixels=100,
    line_width_min_pixels=0,
    get_position="coordinates",
    get_fill_color=[14, 99, 5],
    get_line_color=[0, 0, 0],
)]
# Render
r = pdk.Deck(layers=layer, initial_view_state=view_state, tooltip={"text": "{city}"})
st.pydeck_chart(r)
#####################################################################################################
#User input
with st.sidebar:
    st.dataframe(df3)
    user_solution = [int(0)] + st.multiselect("Select the solution from here", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
#####################################################################################################
#Output
st.write(""" ##### You can compare your results with those of an optimization algorithm used for this problem called Genetic algorithm""")
#User Part
user_solution_array = np.array(user_solution)
st.write("""
 Your solution :
""",str(list(user_solution_array)))
st.write("""
 Your score :
""",fitness(user_solution,cityList)*coef,"Km")
#Algorithm Part
#####################################################################################################
#Algorithm Part
route = geneticAlgorithm(cityList,0.01,100)
st.write("Best solution found by the algorithm :",str(list(route)))
st.write("Score:",fitness(route,cityList)*coef,"Km")
#####################################################################################################
# Data frame creation
# User Solution
x1 = list(x[user_solution])
y1 = list(y[user_solution])
user_path = []
for i in range(0,nbr_villes):
    o = [x1[i]]
    o.append(y1[i])
    user_path.append(o)
user_path.append(user_path[0])
# Algo Solution
x2 = list(x[route])
y2 = list(y[route])
algo_path = []
for i in range(0,nbr_villes):
    o = [x2[i]]
    o.append(y2[i])
    algo_path.append(o)
algo_path.append(algo_path[0])
paths = [algo_path]
paths.append(user_path)
d = {'path':paths}
dataframes = pd.DataFrame(d)
dataframes['color'] = ['#17db02','#ed1c24']
dataframes['name'] = ['Algorithm solution','Your Solution']
st.dataframe(dataframes)
#####################################################################################################
#Visualization of the solutions

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

dataframes['color'] = dataframes['color'].apply(hex_to_rgb)
#####################################################################################################
view_state = pdk.ViewState(
    latitude=32.4833,
    longitude=3.6667,
    zoom=3.5
)
layer =[pdk.Layer(
    "ScatterplotLayer",
    df1,
    pickable=True,
    opacity=1,
    stroked=True,
    filled=True,
    radius_scale=10,
    radius_min_pixels=5,
    radius_max_pixels=100,
    line_width_min_pixels=0,
    get_position="coordinates",
    get_fill_color=[255, 140, 0],
    get_line_color=[0, 0, 0],
),
pdk.Layer(
    "ScatterplotLayer",
    df2,
    pickable=True,
    opacity=1,
    stroked=True,
    filled=True,
    radius_scale=10,
    radius_min_pixels=5,
    radius_max_pixels=100,
    line_width_min_pixels=0,
    get_position="coordinates",
    get_fill_color=[14, 99, 5],
    get_line_color=[0, 0, 0],
),
pdk.Layer(
    type='PathLayer',
    data=dataframes,
    pickable=True,
    get_color='color',
    width_scale=20,
    width_min_pixels=2,
    get_path='path',
    get_width=5
)]
# Render
r = pdk.Deck(layers=layer, initial_view_state=view_state, tooltip={"text": "{city}"})
st.pydeck_chart(r)
st.write(""" The genetic algorithm is a method for solving both constrained and unconstrained optimization problems that is based on natural selection, the process that drives biological evolution. The genetic algorithm repeatedly modifies a population of individual solutions. At each step, the genetic algorithm selects individuals from the current population to be parents and uses them to produce the children for the next generation. Over successive generations, the population "evolves" toward an optimal solution""")
#st.image(image)
