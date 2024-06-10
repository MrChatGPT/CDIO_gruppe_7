/*
    cam2web - streaming camera to web

    Copyright (C) 2017, cvsandbox, cvsandbox@gmail.com

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*/

#ifndef XRASPI_CAMERA_CONFIG_HPP
#define XRASPI_CAMERA_CONFIG_HPP

#include "IObjectConfigurator.hpp"
#include "XRaspiCamera.hpp"

// The class is to get/set camera properties
class XRaspiCameraConfig : public IObjectConfigurator
{
public:
    XRaspiCameraConfig( const std::shared_ptr<XRaspiCamera>& camera );

    XError SetProperty( const std::string& propertyName, const std::string& value );
    XError GetProperty( const std::string& propertyName, std::string& value ) const;

    std::map<std::string, std::string> GetAllProperties( ) const;

private:
    std::shared_ptr<XRaspiCamera> mCamera;
};

// The class is to get/set camera properties information - min, max, default, etc.
class XRaspiCameraPropsInfo : public IObjectInformation
{
public:
    XRaspiCameraPropsInfo( const std::shared_ptr<XRaspiCamera>& camera );

    XError GetProperty( const std::string& propertyName, std::string& value ) const;

    std::map<std::string, std::string> GetAllProperties( ) const;

private:
    std::shared_ptr<XRaspiCamera> mCamera;
};

#endif // XRASPI_CAMERA_CONFIG_HPP

