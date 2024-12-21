import { Component, Input } from '@angular/core';
import { RestaurantMetaData } from '../../interfaces/restaurant-meta';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'restaurant',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './restaurant.component.html',
  styleUrl: './restaurant.component.css'
})
export class RestaurantComponent {

  @Input() restaurant: RestaurantMetaData | undefined;
}
