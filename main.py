import sys
import os
from PyQt5 import uic
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (QMainWindow)
import matplotlib.pyplot as plt
import numpy as np
import math
import random

class AlgorithmGeneticalCanonical():
    
    def __init__(self, initialPopulation, maxPopulation, pmi, pmg, pc, precision, xMin, xMax, generations, max, verbose=True):
        self.initialPopulation = initialPopulation
        self.maxPopulation = maxPopulation
        self.pmi = pmi
        self.pmg = pmg
        self.pc = pc
        self.precision = precision
        self.xMin = xMin
        self.xMax = xMax
        self.generations = generations
        self.max = max
        self.verbose = verbose
        
        
    def start(self, poblacion):
        numberOfPoints = self.calculate(self.xMax, self.xMin, self.precision)

        cruzas = []
        cruzas = self.cruza(poblacion, self.pc )
        individuos_before_poda = self.mutacion(cruzas, self.pmi, self.pmg)
        newElements = self.aptitud_function_for_new_elements(individuos_before_poda)
        for element in newElements:
            poblacion.append(element)
        podaValues = self.poda(poblacion)
        return podaValues

   
    
    def calculate_bits(self, calculo_valor):
        bits = math.ceil(math.log(calculo_valor,2))
        return bits
    
    def generate_population(self):
        poblacion = []
        individual = random.sample(range(0, 10), 3)
        for i in individual:
            x = self.aptitud_function(i)
            poblacion.append([self.decimal_to_bits(i), x, self.fx(x)])
        return poblacion
        
    def decimal_to_bits(self, number):
        bits =  format(number, "b")
        if len(bits) == 1: bits = "000" + bits
        elif len(bits) == 2: bits = "00" + bits
        elif len(bits) == 3: bits = "0" + bits
        
        return bits
    
    def binary_to_decimal(self,individuo):
        decimal = 0
        cadena = ""
        for i in range(len(individuo)):
            cadena += str(individuo[i])    
        for posicion, digito_string in enumerate(cadena[::-1]):
            decimal += int(digito_string) * 2 ** posicion
        return(int(decimal), cadena)
    
    def aptitud_function(self, number):
        return self.xMin + (number * self.precision)
    
    def aptitud_function_for_new_elements(self, newElements):
        listE = []
        for element in newElements:
            decimal = self.binary_to_decimal(element)
            x = self.aptitud_function(int(decimal[0]))
            listE.append([(element), x, self.fx(x)])
        return listE

    def fx(self,x):
        return math.sin(x)*math.log(abs(x-1)) - math.sqrt(x + x**2)
    
         
    def calculate(self, xMax, xMin, precision):
        return ((xMax-xMin)/precision) + 1
    
    def cruza(self, padres,p_cruza):
        hijo1_head = ""
        hijo1_tail = ""
        hijo2_head = ""
        hijo2_tail = ""
        hijo1 = ""
        hijo2 = ""
        hijos = []
        
        for i in range(0, len(padres) - 1):
            if i == len(padres) - 1:
                pass
            else:
                for i in range(i + 1, len(padres) - 1):
                    pc = np.random.rand()
                    if pc <= p_cruza:
                        punto_cruza = int (padres[i].__getitem__(0).__len__() / 2)
                        hijo1_head = padres[i].__getitem__(0)[:punto_cruza]
                        hijo1_tail = padres[i+1].__getitem__(0)[punto_cruza:]
                        hijo2_head = padres[i+1].__getitem__(0)[:punto_cruza]
                        hijo2_tail = padres[i].__getitem__(0)[punto_cruza:]
                        hijo1 = hijo1_head +""+ hijo1_tail
                        hijo2 = hijo2_head +""+ hijo2_tail
                        hijos.append(hijo1)
                        hijos.append(hijo2)
        return hijos
    
    def mutacion(self, hijos, pmi, pmg):
        pmi = pmi
        pmg = pmg
        pm = pmi * pmg
        individuos = []
        
        poblacion_final = []
        for i in range(hijos.__len__()):
            numero_aleatorio = [np.random.rand() for i in range(self.calculate_bits(self.calculate(self.xMax, self.xMin, self.precision)))]
            individuo = (hijos[i], numero_aleatorio)
            individuos.append(individuo)

    
        for i in range(hijos.__len__()):
            for j in range(individuos[i].__getitem__(1).__len__()):
                if individuos[i].__getitem__(1)[j] < pm:
                    individuo = list(individuos[i].__getitem__(0))
                    if individuo[j] == "0":
                        individuo[j] = "1"
                        individuoMutado = "".join(individuo)
                        individuos[i] = (individuoMutado, individuos[i].__getitem__(1))
                    else:
                        individuo[j] = "0"
                        individuoMutado = "".join(individuo)
                        individuos[i] = (individuoMutado, individuos[i].__getitem__(1))
                        

        for i in range(individuos.__len__()):            
            poblacion_final.append(individuos[i].__getitem__(0))
        
        return poblacion_final
    
    def ordenar_por_x(self, valores):
        valores_ordenar = []
        valores_completed= []
        for i in range(valores.__len__()):
            valores_ordenar.append(valores.__getitem__(i).__getitem__(1))
            
        valores_ordenados = sorted(valores_ordenar, key = lambda x:float(x), reverse=self.max)
        if self.max:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:float(x), reverse=True)
        else:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:float(x)) 
        for valor in valores_ordenados:
            for va in valores:
                if valor == va[1]:
                    valores_completed.append([va[0], valor, va[2]])
        return valores_completed
    
    def poda(self, array):
        sortedPopulation = self.ordenar_por_x(array)
        for i in sortedPopulation:
            if i.__getitem__(2) > self.maxPopulation:
                sortedPopulation.remove(i)
        sortedPopulation = self.ordenar_valores(sortedPopulation)
        if len(sortedPopulation) > self.maxPopulation:
            while len(sortedPopulation) > self.maxPopulation:
                sortedPopulation.pop()
        return sortedPopulation
        
    def ordenar_valores(self, valores):
        valores_ordenados = []
        valores_ordenar = []
        valoresCompleted = []
        for i in range(valores.__len__()):
            valores_ordenar.append(valores.__getitem__(i).__getitem__(2))
        if self.max:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:float(x), reverse=True)
        else:
            valores_ordenados = sorted(valores_ordenar, key = lambda x:float(x)) 
        
        for valor in valores_ordenados:
            for va in valores:
                if valor == va[2]:
                    valoresCompleted.append([va[0], va[1], valor])
        return valoresCompleted


class View(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("view.ui", self)
        
def main(initialPopulation,maximum_population,pmg,pmi,pc, precision,initial_value,final_value,generation, max):
    generations = []
    algorithm = AlgorithmGeneticalCanonical(initialPopulation,maximum_population,pmi,pmg,pc,precision,initial_value,final_value, generation, max)
    population = algorithm.generate_population()
    betterResults = []
    worseResults = []
    for i in range(generation):
        population = algorithm.start(population)
        generations.append(algorithm.poda(population))
    for element in generations:
        betterResults.append(element[0][2])
        worseResults.append(element[-1][2])
    for i in range(generations.__len__()):
        print({"Generation " + str(i + 1): generations[i]})
    graphic_one(worseResults)
    
    for i in range(len(generations)):
        listX = []
        listY = []
        limX = algorithm.xMax + 2
        for j in range(len(generations[i])):
            listX.append(generations[i].__getitem__(j).__getitem__(1))
            listY.append(generations[i].__getitem__(j).__getitem__(2))
        graphic_per_generation(listX=listX, listY=listY, generation=i, limX=limX)
        
    interfaz.estado.setText("Mejor aptitud: " + str(betterResults[-1]))
    
def f(t):
    square =  t** 2
    return np.arcsin(t + square)
        
def graphic_per_generation(listX, listY, generation, limX):
    plt.title("Generacion: " + str(generation+1))
    x = np.linspace(-1.6175, 0.6175)
    y = np.arcsin(x + x** 2)
    plt.plot(x, y, 'r')
    plt.grid(True)
    plt.scatter(listX, listY)
    os.makedirs(".\img\generacion/", exist_ok=True)
    plt.savefig(".\img\generacion/generacion"+str(generation+1)+".png")
    plt.close()
        
        
def graphic_one(worseIndividuals):
    plt.plot(worseIndividuals, label="Evolución de la aptitud aptitud", color="red", linestyle="-")
    plt.ylabel('Aptitud')
    plt.xlabel('Generación')
    os.makedirs(".\img\historial/", exist_ok=True)
    plt.savefig(".\img\historial/GraficaHistorial.png")
    plt.close()
    
def init():
    run = True
    initialPopulation = int(interfaz.initial_population_input.text())
    maxPopulation = int(interfaz.maximum_population_input.text())
    pmi = float(interfaz.pmi_input.text())
    pmg = float(interfaz.pmg_input.text())
    pc= float(interfaz.pc_input.text())
    precision= float(interfaz.precision_input.text())
    xMin= float(interfaz.initial_value_input.text())
    xMax = float(interfaz.final_value_input.text())
    generations = int(interfaz.generation_input.text())
    max = bool(interfaz.max_value.text())
    min = bool(interfaz.min_value.text())
    
    if(run == True):
        interfaz.estado.setText("")
        isMax = False
        if interfaz.max_value.isChecked():
            isMax = True
        elif interfaz.min_value.isChecked():
            isMax = False
        main(initialPopulation = initialPopulation, maximum_population = maxPopulation, pmg = pmg, pmi = pmi, pc = pc, precision = precision, initial_value = xMin, final_value = xMax, generation = generations, max = isMax)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    interfaz = uic.loadUi("view.ui")
    interfaz.show()
    interfaz.calculate_button.clicked.connect(init)
    sys.exit(app.exec())