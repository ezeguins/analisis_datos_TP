#!/usr/bin/env python
# coding: utf-8

# In[23]:


from geopy.geocoders import Nominatim
import numpy as np
import re
geolocator = Nominatim(user_agent='tp')


# In[50]:


# Latitud y Longitud de la ciudad centrado tomando como referencia el punto medio de Australia
def coordinates(city):
    # Separación de nombre de ciudad por mayúsculas
    if city == 'PearceRAAF': city = 'PearceRaaf'
    lista = re.findall('[A-Z][^A-Z]*', city)
    ciudad = ""
    for palabras in lista:
        ciudad += str(palabras) + " "
    #city_sep = separate(city)
    ciudad = str(ciudad) + ', Australia'
    location = geolocator.geocode(ciudad, timeout=None)
    return np.int64(location.latitude+25), np.int64(location.longitude-133)


# In[51]:


coordinates('Albany')


# In[33]:





# In[ ]:




