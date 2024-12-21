import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { MatInputModule } from '@angular/material/input';
import { CommonModule } from '@angular/common';
import { RestaurantComponent } from './restaurant/restaurant.component';
import { RestaurantMetaData } from '../interfaces/restaurant-meta';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    MatInputModule,
    CommonModule,
    RestaurantComponent
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})

export class AppComponent {
  title = 'restaurant-finder';

  queryResults: RestaurantMetaData[] = [
    {
      city: "Graz",
      name: "Tree Amici",
      photo: "../assets/images/restaurant-types/italian.jpg",
      state: "Styria"
    },
    {
      city: "Vienna",
      name: "McDonalds",
      photo: "../assets/images/restaurant-types/american.jpg",
      state: "Styria"
    },
    {
      city: "Graz",
      name: "Meet",
      photo: "../assets/images/restaurant-types/asian.jpg",
      state: "Styria"
    },
    {
      city: "Graz",
      name: "Don Camillo",
      photo: "../assets/images/restaurant-types/italian.jpg",
      state: "Styria"
    }
  ]; 
}
