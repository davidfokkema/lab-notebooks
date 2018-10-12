import numpy as np
import pylab as plt
from scipy.optimize import curve_fit

import artist


def plot_spectra(Na, Cs):
    Na = Na[:948]
    Cs = Cs[:948]

    plt.figure()
    plt.subplot(121)
    plt.plot(Na)
    plt.subplot(122)
    plt.plot(Cs)


def plot_spectra_artist(background, Na, Cs, Epc):
    background = background[:948]
    Na = Na[:948]
    Cs = Cs[:948]

    data = lambda y: (range(len(y)), y)

    plot = artist.MultiPlot(1, 3, width=r".3\linewidth")
    subplot = plot.get_subplot_at(0, 0)
    subplot.plot(*data(background), mark=None)
    subplot.set_label("achtergrond")
    subplot = plot.get_subplot_at(0, 1)
    subplot.plot(*data(Na), mark=None)
    subplot.set_label("$^{22}$Na")
    subplot = plot.get_subplot_at(0, 2)
    subplot.plot(*data(Cs), mark=None)
    subplot.set_label("$^{137}$Cs")

    plot.show_xticklabels_for_all()
    plot.set_xticklabels_position(0, 1, 'top')
    plot.set_ylimits_for_all(min=0)
    plot.set_yticks_for_all()
    plot.set_xlabel("ADC kanaal")
    plot.set_ylabel("Signaalsterkte")
    plot.save_as_pdf("Na_Cs_background")

    E = arange(len(Na)) * Epc
    plot = artist.MultiPlot(1, 2, width=r".3\linewidth")
    subplot = plot.get_subplot_at(0, 0)
    subplot.plot(E, Na - background, mark=None)
    subplot.set_label("$^{22}$Na")
    subplot.add_pin(r"\SI{.511}{\mega\electronvolt}", 'above', .511, use_arrow=True)
    subplot.add_pin(r"\SI{1.2745}{\mega\electronvolt}", 'above', 1.2745, use_arrow=True, style="pin distance=20pt")
    subplot = plot.get_subplot_at(0, 1)
    subplot.plot(E, Cs - background, mark=None)
    subplot.set_label("$^{137}$Cs")

    plot.show_xticklabels_for_all()
    plot.set_xticklabels_position(0, 1, 'top')
    plot.set_ylimits_for_all(min=0)
    plot.set_yticks_for_all()
    plot.set_xlabel(r"Energie [\si{\mega\electronvolt}]")
    plot.set_ylabel("Signaalsterkte")
    plot.save_as_document("preview")
    plot.save_as_pdf("Na_Cs")


def fit_energy(Na):
    figure()
    plot(Na)
    a, b = 320, 360
    x = arange(a, b)
    y = Na[a:b]
    f = lambda x, N, mu, sigma: N * normpdf(x, mu, sigma)
    popt, pcov = curve_fit(f, x, y, p0=(max(y), mean([a, b]), 1.))
    print "Energy: 511 keV == %f" % popt[1]
   
    figure()
    Epc = .511 / popt[1]
    plot(Epc * arange(len(Na)), Na)
    plot(x * Epc, f(x, *popt))
    ylim(ymin=0)

    return Epc


background = np.genfromtxt('background-10min-650V.TKA')
Na = np.genfromtxt('Na-22-10min-650V.TKA')
Cs = np.genfromtxt('Cs-137-10min-650V.TKA')
#plot_spectra(Na, Cs)
Epc = fit_energy(Na - background)
plot_spectra_artist(background, Na, Cs, Epc)
