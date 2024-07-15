import numpy as np
from scipy import special as sp
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

datosAlmacenamiento, datosBateria  = np.loadtxt("smart.txt")
def calcularVarianza(listaDatos):
    sumatoria = 0
    varianza = 0
    nsamples = listaDatos.shape[0]
    for elemento in listaDatos:
        sumatoria += elemento
    media = sumatoria / nsamples
    for elemento in listaDatos:
        varianza += ((elemento - media)**2)
    varianza = varianza /(nsamples - 1)
    return varianza, media

varianzaAlmacenamiento, mediaAlmacenamiento = calcularVarianza(datosAlmacenamiento)
varianzaBateria, mediaBateria = calcularVarianza(datosBateria)
print(f"la varianza del almacenamiento es {varianzaAlmacenamiento}. La de la bateria es {varianzaBateria}")

#2
dist_h0 = stats.t(df=datosAlmacenamiento.shape[0] - 1)

def calcularPValor(z, varZ, mu0, mediaZ, dist_h0):
    n = z.shape[0]
    desviacion_estandar = (varZ) ** 0.5
    estadisticoZ = np.sqrt(n) * (mu0 - mediaZ) / desviacion_estandar
    if estadisticoZ >= 0:
        p_valor = dist_h0.cdf(-estadisticoZ) * 2 
    else:
        p_valor = dist_h0.cdf(estadisticoZ) * 2
    return p_valor

pts = 1000
mu0 = np.linspace(datosAlmacenamiento.min(), datosAlmacenamiento.max(), pts)
pvalAlmacenamiento = np.zeros(pts)
for k in range(pts):
    pvalAlmacenamiento[k] = calcularPValor(datosAlmacenamiento, varianzaAlmacenamiento, mediaAlmacenamiento, mu0[k], dist_h0)

muAlmacenamiento = mu0[np.argmax(pvalAlmacenamiento)]
plt.plot(mu0, pvalAlmacenamiento)
plt.xlabel(r"$\mu_0^x$")
plt.ylabel("p-valor")
plt.show()
print(f"mux es {muAlmacenamiento}")


#3
aux = varianzaAlmacenamiento ** 0.5
distribucion = stats.norm(loc=muAlmacenamiento, scale=aux)
rangoXs = np.linspace(distribucion.ppf(0.01), distribucion.ppf(0.99), 100)
empirica = ECDF(datosAlmacenamiento)
plt.plot(empirica.x, empirica.y, drawstyle="steps-post", label="Dist. Empirica")
plt.plot(rangoXs, distribucion.cdf(rangoXs), label="normal")
plt.legend()
plt.xlabel("x")
plt.ylabel("Funcion de Distribucion")
plt.show()

dist_h0 = stats.norm
mu0 = np.linspace(datosBateria.min(), datosBateria.max(), pts )
pValBateria = np.zeros(pts)
for k in range(pts):
    pValBateria[k] = calcularPValor(datosBateria, varianzaBateria, mu0[k], mediaBateria, dist_h0)
muBateria = mu0[np.argmax(pValBateria)]
plt.plot(mu0, pValBateria)
plt.xlabel(r"$\mu_0^y$")
plt.ylabel("p-valor")
plt.show()
print(f"El muy es {muBateria}")

#5

distribucion = stats.norm(muBateria, (varianzaBateria ** 0.5))
ys = np.linspace(distribucion.ppf(0.01), distribucion.ppf(0.99), 100)

plt.plot(ys, distribucion.pdf(ys), label="normal")
plt.hist(datosBateria, bins=50, density=True, label="Histograma")
plt.legend()
plt.xlabel("y")
plt.ylabel("Histograma")
plt.show()




def logVerosimilitud(r, x, y, mux, muy, varx, vary):
    n = len(x)
    desviacionEstandarX = np.sqrt(varx)
    desviacionEstandarY = np.sqrt(vary)
    logL = n *  np.log(1/(2*np.pi * desviacionEstandarX * desviacionEstandarY * np.sqrt(1-(r**2))))
    for i in range(n):
        auxA = -1/(2*(1-r ** 2))
        auxB = (((x[i] - mux) ** 2) / (varx ))
        auxC = ((2 * r * (x[i] - mux) * (y[i] - muy)) /(desviacionEstandarX * desviacionEstandarY))
        auxD = (((y[i] - muy) ** 2) / (vary)) 
        logL += auxA * (auxB - auxC + auxD)
    return logL
pts = 1000
rs = np.linspace(-0.9, 0.9, pts)
logL = [logVerosimilitud(r, datosAlmacenamiento, datosBateria, muAlmacenamiento, muBateria, varianzaAlmacenamiento, varianzaBateria) for r in rs]
rho = rs[np.argmax(logL)]
print(f"rho es {rho}")
plt.plot(rs, logL)
plt.xlabel(r"$\rho$")
plt.ylabel(r"$log(L(\rho))$")
plt.show()


#7

def rectaRegresion(x, mux, muy, varx, vary, rho):
    y = ((rho * (varx ** 0.5) * (vary ** 0.5))/varx) * (x - mux) + muy
    return y

print(rectaRegresion(256, muAlmacenamiento, muBateria, varianzaAlmacenamiento, varianzaBateria, rho))
plt.scatter(datosAlmacenamiento, datosBateria, c='red')
plt.plot([0, 256], [rectaRegresion(0, muAlmacenamiento, muBateria, varianzaAlmacenamiento, varianzaBateria, rho), rectaRegresion(256, muAlmacenamiento, muBateria, varianzaAlmacenamiento, varianzaBateria, rho)], 'b')
plt.xlabel("Capacidad de almacenamiento")
plt.ylabel("Duracion de las baterias")
plt.show()