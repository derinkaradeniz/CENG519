from ast import For
from cmath import sqrt
from eva import EvaProgram, Input, Output, evaluate, Expr, py_to_eva, _py_to_term, _curr
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import timeit
import networkx as nx
from random import random
import queue
import math

def generateGraph(n, k, p):
    ws = nx.watts_strogatz_graph(n,k,p)
    return ws

def serializeGraphZeroOne(GG,vec_size):
    n = GG.size()
    graphdict = {}
    g = []
    for row in range(n):
        for column in range(n):
            if GG.has_edge(row, column) or row==column: 
                weight = 1
            else:
                weight = 0 
            g.append( weight  )  
            key = str(row)+'-'+str(column)
            graphdict[key] = [weight] # EVA requires str:listoffloat
    # EVA vector size has to be large, if the vector representation of the graph is smaller, fill the eva vector with zeros
    for i in range(vec_size - n*n): 
        g.append(0.0)
    return g, graphdict

# To display the generated graph
def printGraph(graph,n):
    for row in range(n):
        for column in range(n):
            print("{:.5f}".format(graph[row*n+column]), end = '\t')
        print() 

# Eva requires special input, this function prepares the eva input
# Eva will then encrypt them
def prepareInput(n, m):
    input = {}
    GG = generateGraph(n,3,0.5)
    graph, graphdict = serializeGraphZeroOne(GG,m)
    printGraph(graph,n)
    input['Graph'] = graph
    return input
    
def approxSgnFunc(x,n):
    res = 0

    powX = (1-(x*x))
    for i in range(0,n):
        powX = powX * powX

    res = res + ( (1/(math.pow(4,n))) \
        * math.comb(2*n,n) \
        * x \
        * powX )
    
    return powX

def newComp(a,b,n,d):
    x = a - b
    i = 1
    for i in range(d):
        x = approxSgnFunc(x,n)

    return (x+1)*(1/2)

def graphMask(k, vecSize):
    mask=[0.0]*vecSize
    mask[i] = 1.0
    return mask
            
def revealEncGraph(encGraph,nodeCount,vecSize):
    graphArraySize = nodeCount * nodeCount
    zeros = [0.0] * vecSize
    realGraph = [0.0] * vecSize
    zeroEncGraph = encGraph * zeros
    
    for i in range(graphArraySize):
        maskedEncGraph = encGraph * graphMask(i,vecSize)  
        if newComp(maskedEncGraph, zeroEncGraph) != 1.0:
            realGraph[i] = 1.0      
    
    return realGraph

def bfsWithDistance(g, mark, u, dis, nodeCount):
    lastMarked = 0
    q = queue.Queue()
    
    q.put(u)
    dis[u] = 0
    
    while (not q.empty()):
        u = q.get()
        
        if (mark[u]):
            lastMarked = u
            
        for i in range(nodeCount):
            isNeighbour = g[u*nodeCount + i]
            
            if isNeighbour == 1: 
                
                if (dis[i] == -1):
                    dis[i] = dis[u] + 1
                    q.put(i)
                    
    return lastMarked

def nodesKDistanceFromMarked(encGraph,V, marked, K):    
    g = revealEncGraph(encGraph,V,vecSize=4096*4)
    
    mark = [False] * V
    for i in range(len(marked)):
        mark[marked[i]] = True
 
    # vectors to store distances
    tmp = [-1] * V
    dl = [-1] * V
    dr = [-1] * V
 
    u = bfsWithDistance(g, mark, 0, tmp, V)
    
    u = bfsWithDistance(g, mark, u, dl, V)
    
    bfsWithDistance(g, mark, u, dr, V)
 
    res = 0
    for i in range(V):
        if (dl[i] <= K and dr[i] <= K):
            res += 1
    return res

def graphanalticprogram(graph,nodeCount,marked,kDistance):    
    #marked = [1,3,5]
    #nodeCount = 10
    #kDistance = 3
    reval = nodesKDistanceFromMarked(graph, nodeCount, marked, kDistance)
    
    return reval
    
class EvaProgramDriver(EvaProgram):
    def __init__(self, name, vec_size=4096, n=4,markedNodes=[1,3,5],kDistance=3):
        self.n = n
        self.kDistance = kDistance
        self.markedNodes = markedNodes
        super().__init__(name, vec_size)

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)


def simulate(n,markedNodes,kDistance):
    m = 4096*4
    markedNodes=[1,3,5]
    print("Will start simulation for ", n)
    config = {}
    config['warn_vec_size'] = 'false'
    config['lazy_relinearize'] = 'true'
    config['rescaler'] = 'always'
    config['balance_reductions'] = 'true'
    inputs = prepareInput(n, m)

    graphanaltic = EvaProgramDriver("graphanaltic", vec_size=m,n=n,markedNodes=markedNodes,kDistance=kDistance)
    with graphanaltic:
        graph = Input('Graph')
        reval = graphanalticprogram(graph,n,markedNodes,kDistance)
        Output('ReturnedValue', reval)
    
    prog = graphanaltic
    prog.set_output_ranges(30)
    prog.set_input_scales(30)

    start = timeit.default_timer()
    compiler = CKKSCompiler(config=config)
    compiled_multfunc, params, signature = compiler.compile(prog)
    compiletime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    public_ctx, secret_ctx = generate_keys(params)
    keygenerationtime = (timeit.default_timer() - start) * 1000.0 #ms
    
    start = timeit.default_timer()
    encInputs = public_ctx.encrypt(inputs, signature)
    encryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
    executiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    outputs = secret_ctx.decrypt(encOutputs, signature)
    decryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    reference = evaluate(compiled_multfunc, inputs)
    referenceexecutiontime = (timeit.default_timer() - start) * 1000.0 #ms
    
    # Change this if you want to output something or comment out the two lines below
    for key in outputs:
        print(key, float(outputs[key][0]), float(reference[key][0]))

    mse = valuation_mse(outputs, reference) # since CKKS does approximate computations, this is an important measure that depicts the amount of error

    return compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse


if __name__ == "__main__":
    simcnt = 3 #The number of simulation runs, set it to 3 during development otherwise you will wait for a long time
    # For benchmarking you must set it to a large number, e.g., 100
    #Note that file is opened in append mode, previous results will be kept in the file
    resultfile = open("results.csv", "a")  # Measurement results are collated in this file for you to plot later on
    resultfile.write("NodeCount,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
    resultfile.close()
    
    print("Simulation campaing started:")
    
    n=10
    i=1
    compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(n,[1,3,5],3)
    res = str(n) + "," + str(i) + "," + str(compiletime) + "," + str(keygenerationtime) + "," +  str(encryptiontime) + "," +  str(executiontime) + "," +  str(decryptiontime) + "," +  str(referenceexecutiontime) + "," +  str(mse) + "\n"
    print(res)